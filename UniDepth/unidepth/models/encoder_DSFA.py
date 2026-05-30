from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from unidepth.models.backbones import ConvNeXt, ConvNeXtV2, _make_dinov2_model


class ModelWrap(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.backbone = model

    def forward(self, x, *args, **kwargs):
        features = []
        for layer in self.backbone.features:
            x = layer(x)
            features.append(x)
        return features


def convnextv2_base(config, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs,
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_384_ema.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=False
    )["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnextv2_large(config, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs,
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=False
    )["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnextv2_large_mae(config, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs,
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_large_1k_224_fcmae.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=False
    )["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnextv2_huge(config, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[352, 704, 1408, 2816],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs,
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_512_ema.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=False
    )["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnextv2_huge_mae(config, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[352, 704, 1408, 2816],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs,
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_huge_1k_224_fcmae.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=False
    )["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnext_large_pt(config, **kwargs):
    model = ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs,
    )
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import disable_progress_bars

    from unidepth.models.backbones.convnext import HF_URL, checkpoint_filter_fn

    disable_progress_bars()
    repo_id, filename = HF_URL["convnext_large_pt"]
    state_dict = torch.load(hf_hub_download(repo_id=repo_id, filename=filename))
    state_dict = checkpoint_filter_fn(state_dict, model)
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnext_large(config, **kwargs):
    model = ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        **kwargs,
    )
    return model


def dinov2_vits14(config, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    vit = _make_dinov2_model(
        arch_name="vit_small",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [3, 6, 9, 12]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        export=config.get("export", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        **kwargs,
    )
    return vit


def dinov2_vitb14(config, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-B/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    vit = _make_dinov2_model(
        arch_name="vit_base",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [3, 6, 9, 12]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        export=config.get("export", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        **kwargs,
    )
    return vit


def dinov2_vitl14(config, pretrained: str = "", **kwargs):
    """
    DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    vit = _make_dinov2_model(
        arch_name="vit_large",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [5, 12, 18, 24]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        export=config.get("export", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        **kwargs,
    )
    return vit


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention with optional SDPA backend."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = dropout
        self.use_sdpa = use_sdpa

    def forward(
        self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(
            bsz, seq_len, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        if self.use_sdpa and attn_bias is None:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop if self.training else 0.0,
                is_causal=False,
            )
        else:
            scores = q @ k.transpose(-2, -1)
            if attn_bias is not None:
                if attn_bias.dim() == 3:
                    attn_bias = attn_bias.unsqueeze(1)
                scores = scores + attn_bias
            attn = scores.softmax(dim=-1)
            attn = F.dropout(attn, p=self.attn_drop, training=self.training)
            attn_out = attn @ v

        attn_out = (
            attn_out.transpose(1, 2)
            .reshape(bsz, seq_len, self.embed_dim)
            .contiguous()
        )
        return self.out_proj(attn_out)


class DividedSpaceFocusBlock(nn.Module):
    """Apply per-frame spatial attention followed by per-patch focus attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        layerscale_init: float = 0.1,
        alibi_scale: float = 1.0,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.alibi_scale = alibi_scale

        # Spatial branch (frame-wise)
        self.ln_space_attn = nn.LayerNorm(embed_dim)
        self.space_attn = MultiHeadSelfAttention(
            embed_dim, num_heads=num_heads, dropout=dropout, use_sdpa=use_sdpa
        )
        self.gamma_space_attn = nn.Parameter(
            layerscale_init * torch.ones(embed_dim)
        )
        self.ln_space_ffn = nn.LayerNorm(embed_dim)
        self.space_ffn = MLP(embed_dim, mlp_ratio=4.0, dropout=dropout)
        self.gamma_space_ffn = nn.Parameter(
            layerscale_init * torch.ones(embed_dim)
        )

        # Focus branch (cross-frame per patch)
        self.ln_focus_attn = nn.LayerNorm(embed_dim)
        self.focus_attn = MultiHeadSelfAttention(
            embed_dim, num_heads=num_heads, dropout=dropout, use_sdpa=use_sdpa
        )
        self.gamma_focus_attn = nn.Parameter(
            layerscale_init * torch.ones(embed_dim)
        )
        self.ln_focus_ffn = nn.LayerNorm(embed_dim)
        self.focus_ffn = MLP(embed_dim, mlp_ratio=4.0, dropout=dropout)
        self.gamma_focus_ffn = nn.Parameter(
            layerscale_init * torch.ones(embed_dim)
        )

        hidden = max(1, embed_dim // 4)
        self.k_embed = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, embed_dim),
        )
        self.focus_film = nn.Linear(embed_dim, embed_dim * 2)

    def _residual(
        self, x: torch.Tensor, dx: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        return x + dx * gamma if gamma is not None else x + dx

    def _space_attention(
        self, x: torch.Tensor, k_tokens: Optional[torch.Tensor]
    ) -> torch.Tensor:
        bsz, num_frames, num_patches, dim = x.shape
        seq = x.reshape(bsz * num_frames, num_patches, dim)
        if k_tokens is not None:
            k_tok = k_tokens.reshape(bsz * num_frames, 1, dim)
            seq = torch.cat([k_tok, seq], dim=1)

        h = self.space_attn(self.ln_space_attn(seq))
        seq = self._residual(seq, h, self.gamma_space_attn)

        h2 = self.space_ffn(self.ln_space_ffn(seq))
        seq = self._residual(seq, h2, self.gamma_space_ffn)

        if k_tokens is not None:
            seq = seq[:, 1:, :]

        return seq.reshape(bsz, num_frames, num_patches, dim)

    def _focus_attention(
        self,
        x: torch.Tensor,
        k_values: Optional[torch.Tensor],
        film_scale: Optional[torch.Tensor],
        film_shift: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, num_frames, num_patches, dim = x.shape

        if film_scale is not None and film_shift is not None:
            scale = torch.tanh(film_scale).unsqueeze(2)
            shift = film_shift.unsqueeze(2)
            x = x * (1 + scale) + shift

        if k_values is not None:
            diff = torch.abs(k_values[:, :, None] - k_values[:, None, :])
            bias = -self.alibi_scale * diff.squeeze(-1)
            bias = bias.unsqueeze(1).repeat(1, num_patches, 1, 1)
            bias = bias.reshape(bsz * num_patches, num_frames, num_frames)
            bias = bias.to(dtype=x.dtype)
        else:
            bias = None

        x_perm = x.permute(0, 2, 1, 3).reshape(bsz * num_patches, num_frames, dim)

        h = self.focus_attn(self.ln_focus_attn(x_perm), attn_bias=bias)
        x_perm = self._residual(x_perm, h, self.gamma_focus_attn)

        h2 = self.focus_ffn(self.ln_focus_ffn(x_perm))
        x_perm = self._residual(x_perm, h2, self.gamma_focus_ffn)

        return (
            x_perm.reshape(bsz, num_patches, num_frames, dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def forward(
        self,
        ref_patches: torch.Tensor,
        stack_patches: Optional[torch.Tensor],
        k_stack: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if stack_patches is None or stack_patches.numel() == 0:
            return ref_patches

        bsz, num_patches, _ = ref_patches.shape
        num_frames = stack_patches.shape[1] + 1

        x = torch.cat(
            [ref_patches.unsqueeze(1), stack_patches], dim=1
        )  # [B, F, K, D]

        device = ref_patches.device
        dtype = ref_patches.dtype

        if k_stack is None:
            k_stack = torch.zeros(
                bsz,
                num_frames - 1,
                1,
                device=device,
                dtype=dtype,
            )
        elif k_stack.dim() == 2:
            k_stack = k_stack.unsqueeze(-1)

        zeros = torch.zeros(bsz, 1, 1, device=device, dtype=k_stack.dtype)
        k_values = torch.cat([zeros, k_stack], dim=1)
        k_tokens = self.k_embed(k_values.to(dtype=self.k_embed[0].weight.dtype))
        film_scale, film_shift = self.focus_film(k_tokens).chunk(2, dim=-1)
        film_scale = film_scale.to(dtype=ref_patches.dtype)
        film_shift = film_shift.to(dtype=ref_patches.dtype)

        x = self._space_attention(x, k_tokens)
        x = self._focus_attention(x, k_values, film_scale, film_shift)

        return x[:, 0]


class DSFADinov2Encoder(nn.Module):
    """DINOv2 encoder with Divided Space-Focus Attention fusion."""

    def __init__(self, config: dict, arch_name: str = "vit_large", **kwargs) -> None:
        super().__init__()
        output_idx = config.get("output_idx", [6, 12, 18, 24])
        backbone_kwargs = dict(
            arch_name=arch_name,
            pretrained=config.get("pretrained", ""),
            output_idx=output_idx,
            checkpoint=config.get("use_checkpoint", False),
            drop_path_rate=config.get("drop_path", 0.0),
            num_register_tokens=config.get("num_register_tokens", 0),
            use_norm=config.get("use_norm", False),
            export=config.get("export", False),
            interpolate_offset=config.get("interpolate_offset", 0.0),
            frozen_stages=config.get("frozen_stages", 0),
        )
        backbone_kwargs.update(kwargs)
        self.backbone = _make_dinov2_model(**backbone_kwargs)

        self.embed_dim = getattr(self.backbone, "embed_dim", self.backbone.num_features)
        self.embed_dims = getattr(self.backbone, "embed_dims", [self.embed_dim])
        self.depths = list(getattr(self.backbone, "depths", output_idx))
        self.cls_token_embed_dims = getattr(
            self.backbone,
            "cls_token_embed_dims",
            self.embed_dims,
        )
        self.patch_size = getattr(self.backbone, "patch_size", 14)

        num_stages = len(self.depths)
        default_layers: List[int] = []
        if num_stages >= 2:
            default_layers.append(num_stages - 2)
        if num_stages >= 1:
            default_layers.append(num_stages - 1)
        fusion_layers_cfg = config.get("fusion_layers", default_layers)
        fusion_layers = [
            int(idx) for idx in fusion_layers_cfg if 0 <= int(idx) < num_stages
        ]
        self.fusion_layers = sorted(set(fusion_layers))

        fusion_heads = config.get("fusion_num_heads", config.get("num_heads", 8))
        fusion_dropout = config.get("fusion_dropout", config.get("attn_dropout", 0.0))
        layerscale_init = config.get("fusion_layerscale", config.get("layer_scale", 0.1))
        alibi_scale = config.get("alibi_scale", 1.0)
        use_sdpa = config.get("use_sdpa", True)

        self.stage_ranges = list(zip([0, *self.depths[:-1]], self.depths))
        self.fusion_blocks = nn.ModuleDict()
        for stage_idx in self.fusion_layers:
            self.fusion_blocks[str(stage_idx)] = DividedSpaceFocusBlock(
                embed_dim=self.embed_dim,
                num_heads=fusion_heads,
                dropout=fusion_dropout,
                layerscale_init=layerscale_init,
                alibi_scale=alibi_scale,
                use_sdpa=use_sdpa,
            )

    def forward(
        self,
        x: torch.Tensor,
        focus_stack: Optional[torch.Tensor] = None,
        focus_k: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        ref_blocks, ref_cls = self.backbone(x)

        if (
            not self.fusion_blocks
            or focus_stack is None
            or focus_stack.numel() == 0
        ):
            return ref_blocks, ref_cls

        if focus_stack.dim() != 5:
            raise ValueError(
                f"focus_stack must be a 5D tensor [B, N, C, H, W], got {focus_stack.shape}"
            )

        focus_stack = focus_stack.to(dtype=x.dtype, device=x.device)
        batch, stack_len = focus_stack.shape[:2]
        frames = focus_stack.reshape(batch * stack_len, *focus_stack.shape[2:])

        stack_blocks, stack_cls = self.backbone(frames)

        stack_blocks = [
            feat.reshape(batch, stack_len, *feat.shape[1:]) for feat in stack_blocks
        ]
        stack_cls = [
            tok.reshape(batch, stack_len, *tok.shape[1:]) for tok in stack_cls
        ]

        if focus_k is not None:
            focus_k = focus_k.to(dtype=x.dtype, device=x.device)
            if focus_k.dim() == 1:
                focus_k = focus_k.unsqueeze(0)
            if focus_k.dim() == 2:
                focus_k = focus_k.unsqueeze(-1)
            if focus_k.dim() != 3 or focus_k.shape[1] != stack_len:
                raise ValueError(
                    f"focus_k must have shape [B, N] or [B, N, 1], got {focus_k.shape}"
                )
        else:
            focus_k = None

        fused_blocks: List[torch.Tensor] = list(ref_blocks)

        for stage_idx, (start, end) in enumerate(self.stage_ranges):
            module_key = str(stage_idx)
            if module_key not in self.fusion_blocks:
                continue

            block_index = end - 1
            ref_feature = fused_blocks[block_index]
            stack_feature = stack_blocks[block_index]

            bsz, h, w, dim = ref_feature.shape
            ref_tokens = ref_feature.reshape(bsz, h * w, dim)
            stack_tokens = stack_feature.reshape(bsz, stack_len, h * w, dim)

            refined = self.fusion_blocks[module_key](
                ref_patches=ref_tokens,
                stack_patches=stack_tokens,
                k_stack=focus_k,
            )
            refined_feature = refined.view(bsz, h, w, dim).contiguous()

            for blk_idx in range(start, end):
                fused_blocks[blk_idx] = refined_feature

        return fused_blocks, ref_cls

    def get_params(self, lr, wd, ld, *args, **kwargs):
        encoder_groups, encoder_lr = self.backbone.get_params(
            lr, wd, ld, *args, **kwargs
        )
        fusion_params = [
            param
            for param in self.fusion_blocks.parameters()
            if param.requires_grad
        ]
        if fusion_params:
            encoder_groups.append(
                {
                    "params": fusion_params,
                    "lr": lr,
                    "weight_decay": wd,
                }
            )
            encoder_lr.append(lr)
        return encoder_groups, encoder_lr


def dinov2_vitl14_DSFA(config, **kwargs):
    return DSFADinov2Encoder(config, arch_name="vit_large", **kwargs)
