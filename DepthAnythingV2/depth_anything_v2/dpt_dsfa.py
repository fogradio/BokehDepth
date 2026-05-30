import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .dpt import DPTHead
from .util.transform import Resize, NormalizeImage, PrepareForNet


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Lightweight MSA wrapper that supports SDPA fallback and additive masks."""

    def __init__(self, embed_dim, num_heads=8, dropout=0.0, use_sdpa=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = dropout
        self.use_sdpa = use_sdpa

    def forward(self, x, attn_bias=None):
        # x: [B, L, D]
        B, L, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, L, head_dim]
        q = q * self.scale

        if self.use_sdpa and (attn_bias is None):
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop if self.training else 0.0,
                is_causal=False,
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1))
            if attn_bias is not None:
                # attn_bias: [B, L, L] or broadcastable to that
                if attn_bias.dim() == 3:
                    attn_bias = attn_bias.unsqueeze(1)
                scores = scores + attn_bias
            attn = scores.softmax(dim=-1)
            attn = F.dropout(attn, p=self.attn_drop, training=self.training)
            attn_out = torch.matmul(attn, v)

        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.embed_dim)
        return self.out_proj(attn_out)


class DividedSpaceFocusBlock(nn.Module):
    """DSFA block: per-frame space attention followed by per-patch focus attention."""

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        dropout=0.0,
        layerscale_init=0.1,
        alibi_scale=1.0,
        use_sdpa=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.alibi_scale = alibi_scale

        # Space branch (frame wise)
        self.ln_space_attn = nn.LayerNorm(embed_dim)
        self.space_attn = MultiHeadSelfAttention(embed_dim, num_heads=num_heads, dropout=dropout, use_sdpa=use_sdpa)
        self.gamma_space_attn = nn.Parameter(layerscale_init * torch.ones(embed_dim))
        self.ln_space_ffn = nn.LayerNorm(embed_dim)
        self.space_ffn = MLP(embed_dim, mlp_ratio=4.0, dropout=dropout)
        self.gamma_space_ffn = nn.Parameter(layerscale_init * torch.ones(embed_dim))

        # Focus branch (cross-frame per patch)
        self.ln_focus_attn = nn.LayerNorm(embed_dim)
        self.focus_attn = MultiHeadSelfAttention(embed_dim, num_heads=num_heads, dropout=dropout, use_sdpa=use_sdpa)
        self.gamma_focus_attn = nn.Parameter(layerscale_init * torch.ones(embed_dim))
        self.ln_focus_ffn = nn.LayerNorm(embed_dim)
        self.focus_ffn = MLP(embed_dim, mlp_ratio=4.0, dropout=dropout)
        self.gamma_focus_ffn = nn.Parameter(layerscale_init * torch.ones(embed_dim))

        # K conditioning
        self.k_embed = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.ReLU(True),
            nn.Linear(embed_dim // 4, embed_dim),
        )
        self.focus_film = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
        )

    def _residual(self, x, dx, gamma):
        return x + dx * gamma if gamma is not None else x + dx

    def _space_attention(self, x, k_tokens):
        # x: [B, F, K, D], k_tokens: [B, F, D]
        B, F, K, D = x.shape
        seq = x.reshape(B * F, K, D)
        k_tok = k_tokens.reshape(B * F, 1, D)
        seq = torch.cat([k_tok, seq], dim=1)

        h = self.ln_space_attn(seq)
        h = self.space_attn(h)
        seq = self._residual(seq, h, self.gamma_space_attn)

        h2 = self.ln_space_ffn(seq)
        h2 = self.space_ffn(h2)
        seq = self._residual(seq, h2, self.gamma_space_ffn)

        seq = seq[:, 1:, :]
        return seq.reshape(B, F, K, D)

    def _focus_attention(self, x, k_values, film_scale, film_shift):
        # x: [B, F, K, D]
        B, F, K, D = x.shape

        # Apply FiLM modulation with K
        scale = torch.tanh(film_scale).unsqueeze(2)  # [B, F, 1, D]
        shift = film_shift.unsqueeze(2)  # [B, F, 1, D]
        x = x * (1 + scale) + shift

        # Build ALiBi-like bias using K values
        # k_values: [B, F, 1]
        diff = torch.abs(k_values[:, :, None] - k_values[:, None, :])  # [B, F, F, 1]
        bias = -self.alibi_scale * diff.squeeze(-1)  # [B, F, F]

        x_perm = x.permute(0, 2, 1, 3).reshape(B * K, F, D)
        # broadcast bias for each spatial location
        bias_expanded = bias.unsqueeze(1).repeat(1, K, 1, 1).reshape(B * K, F, F).to(x_perm.dtype)

        h = self.ln_focus_attn(x_perm)
        h = self.focus_attn(h, attn_bias=bias_expanded)
        x_perm = self._residual(x_perm, h, self.gamma_focus_attn)

        h2 = self.ln_focus_ffn(x_perm)
        h2 = self.focus_ffn(h2)
        x_perm = self._residual(x_perm, h2, self.gamma_focus_ffn)

        return x_perm.reshape(B, K, F, D).permute(0, 2, 1, 3)

    def forward(self, ref_patches, stack_patches, k_stack):
        # ref_patches: [B, K, D], stack_patches: [B, N, K, D], k_stack: [B, N, 1]
        B, K, D = ref_patches.shape
        if stack_patches is None or stack_patches.numel() == 0:
            return ref_patches

        N = stack_patches.shape[1]
        device = ref_patches.device
        if k_stack is None:
            k_stack = torch.zeros(B, N, 1, device=device, dtype=ref_patches.dtype)
        elif k_stack.dim() == 2:
            k_stack = k_stack.unsqueeze(-1)

        ref = ref_patches.unsqueeze(1)
        x = torch.cat([ref, stack_patches], dim=1)  # [B, N+1, K, D]

        zeros = torch.zeros(B, 1, 1, device=device, dtype=k_stack.dtype)
        k_values = torch.cat([zeros, k_stack], dim=1)  # [B, N+1, 1]
        k_tokens = self.k_embed(k_values.to(dtype=self.k_embed[0].weight.dtype))  # [B, N+1, D]
        film_scale, film_shift = self.focus_film(k_tokens).chunk(2, dim=-1)
        k_tokens = k_tokens.to(dtype=ref_patches.dtype)
        film_scale = film_scale.to(dtype=ref_patches.dtype)
        film_shift = film_shift.to(dtype=ref_patches.dtype)

        x = self._space_attention(x, k_tokens)
        x = self._focus_attention(x, k_values, film_scale, film_shift)

        return x[:, 0]


class DSFAEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=8,
        dropout=0.0,
        layerscale_init=0.1,
        alibi_scale=1.0,
        use_sdpa=True,
    ):
        super().__init__()
        self.block = DividedSpaceFocusBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            layerscale_init=layerscale_init,
            alibi_scale=alibi_scale,
            use_sdpa=use_sdpa,
        )

    def forward(self, ref_patches, stack_patches=None, k_stack=None):
        return self.block(ref_patches, stack_patches, k_stack)


class EncoderFusionModule(nn.Module):
    """Apply DSFA blocks at selected ViT intermediate layers."""

    def __init__(
        self,
        embed_dim,
        fusion_layers=(2, 3),
        num_heads=8,
        dropout=0.0,
        layerscale_init=0.1,
        alibi_scale=1.0,
        use_sdpa=True,
    ):
        super().__init__()
        self.fusion_layers = list(fusion_layers)
        self.layers = nn.ModuleDict()
        for idx in self.fusion_layers:
            self.layers[str(idx)] = DSFAEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                layerscale_init=layerscale_init,
                alibi_scale=alibi_scale,
                use_sdpa=use_sdpa,
            )

    def forward(self, features, stack_features, k_stack, patch_h, patch_w):
        if stack_features is None:
            return features

        fused = []

        for i, (patch_tokens, cls_token) in enumerate(features):
            if i in self.fusion_layers and str(i) in self.layers:
                stack_patch_tokens, _ = stack_features[i]
                refined = self.layers[str(i)](
                    ref_patches=patch_tokens,
                    stack_patches=stack_patch_tokens,
                    k_stack=k_stack,
                )
                fused.append((refined, cls_token))
            else:
                fused.append((patch_tokens, cls_token))
        return fused


class DepthAnythingV2DSFA(nn.Module):
    """Depth Anything V2 with Divided Space-Focus Attention backbone fusion."""

    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        max_depth=20.0,
        fusion_layers=(2, 3),
        grad_free_stack=False,
        num_heads=8,
        attn_dropout=0.0,
        layerscale_init=0.1,
        alibi_scale=1.0,
        use_sdpa=True,
    ):
        super().__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }

        self.max_depth = max_depth
        self.encoder = encoder
        self.fusion_layers = list(fusion_layers)
        self.grad_free_stack = grad_free_stack

        self.pretrained = DINOv2(model_name=encoder, use_sdpa=use_sdpa)

        self.encoder_fusion = EncoderFusionModule(
            embed_dim=self.pretrained.embed_dim,
            fusion_layers=fusion_layers,
            num_heads=num_heads,
            dropout=attn_dropout,
            layerscale_init=layerscale_init,
            alibi_scale=alibi_scale,
            use_sdpa=use_sdpa,
        )

        self.depth_head = DPTHead(
            self.pretrained.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
        )

    def forward(self, x, focus_stack=None, k_stack=None):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        features = self.pretrained.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder], return_class_token=True
        )

        if focus_stack is not None and k_stack is not None:
            _, N = focus_stack.shape[:2]
            stack_features = [[] for _ in range(len(features))]
            for i in range(N):
                frame = focus_stack[:, i]
                if self.grad_free_stack and self.training:
                    with torch.no_grad():
                        frame_feats = self.pretrained.get_intermediate_layers(
                            frame, self.intermediate_layer_idx[self.encoder], return_class_token=True
                        )
                else:
                    frame_feats = self.pretrained.get_intermediate_layers(
                        frame, self.intermediate_layer_idx[self.encoder], return_class_token=True
                    )
                for j, feat in enumerate(frame_feats):
                    stack_features[j].append(feat)

            for j in range(len(stack_features)):
                patch_tokens_list = []
                cls_tokens_list = []
                for i in range(N):
                    patch_tok, cls_tok = stack_features[j][i]
                    patch_tokens_list.append(patch_tok)
                    cls_tokens_list.append(cls_tok)
                stacked_patch = torch.stack(patch_tokens_list, dim=1)
                stacked_cls = torch.stack(cls_tokens_list, dim=1) if cls_tokens_list[0] is not None else None
                stack_features[j] = (stacked_patch, stacked_cls)

            features = self.encoder_fusion(features, stack_features, k_stack, patch_h, patch_w)

        depth = self.depth_head(features, patch_h, patch_w) * self.max_depth
        return depth.squeeze(1)

    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518, focus_stack=None, k_stack=None):
        image, (h, w) = self.image2tensor(raw_image, input_size)

        focus_tensor = None
        k_tensor = None
        if focus_stack is not None and k_stack is not None:
            focus_tensors = []
            for frame in focus_stack:
                frame_tensor, _ = self.image2tensor(frame, input_size)
                focus_tensors.append(frame_tensor[0])
            focus_tensor = torch.stack(focus_tensors, dim=0).unsqueeze(0)

            k_tensor = torch.tensor(k_stack, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            k_tensor = k_tensor.to(image.device)

        depth = self.forward(image, focus_tensor, k_tensor)
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        return depth.cpu().numpy()

    def image2tensor(self, raw_image, input_size=518):
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        h, w = raw_image.shape[:2]
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(device)
        return image, (h, w)
