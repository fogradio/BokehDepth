import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import logging  # logging support
import os  # path handling
from diffusers.models.activations import get_activation
from diffusers.models.embeddings import get_2d_sincos_pos_embed
from safetensors.torch import load_file as load_safetensors  # safetensors format support


def get_simple_embedding_layer(sequence_length, output_dim, hidden_dim=64, activation='relu'):
    """Two-layer MLP that lifts the scalar bokeh-strength input into the
    cross-attention context dimension. Inlined from the (now deleted)
    bokeh_adapter_sd module since this is the only piece we still need on
    the Flux base model.
    """
    return nn.Sequential(
        nn.Linear(sequence_length, hidden_dim),
        get_activation(activation),
        nn.Linear(hidden_dim, output_dim),
    )


def _pick_sdpa_mask(attention_mask, query, key, num_heads, logger=None):
    """
    Pick an SDPA-compatible mask out of the ``attention_mask`` that diffusers
    0.35+ may hand us (which can be a Tensor or a tuple):
      - 4D: (..., Lq, Lk) -- used directly
      - 3D: (B*H, Lq, Lk) -- used directly
      - 2D: (Lq, Lk) -- used directly
    If nothing matches we return None and emit a warning.
    """
    if attention_mask is None:
        return None

    if torch.is_tensor(attention_mask):
        return attention_mask  # let SDPA handle broadcasting / validation

    if isinstance(attention_mask, tuple):
        Lq = query.shape[2]
        Lk = key.shape[2]
        cands = []

        for i, t in enumerate(attention_mask):
            if t is None or not torch.is_tensor(t):
                continue
            if t.dim() == 4 and t.shape[-2] == Lq and t.shape[-1] == Lk:
                cands.append((t, i, "4D"))
            elif t.dim() == 3 and t.shape[-2] == Lq and t.shape[-1] == Lk:
                cands.append((t, i, "3D"))
            elif t.dim() == 2 and t.shape[-2] == Lq and t.shape[-1] == Lk:
                cands.append((t, i, "2D"))

        if cands:
            selected_mask, idx, dim_info = cands[0]
            if logger:
                logger.debug(f"picked tuple index {idx} ({dim_info} mask), shape: {tuple(selected_mask.shape)}")

            # If the key was concatenated and Lk grew, pad the mask to match.
            if selected_mask.shape[-1] != Lk:
                pad_length = Lk - selected_mask.shape[-1]
                if pad_length > 0:
                    selected_mask = F.pad(selected_mask, (0, pad_length), value=0.0)
                    if logger:
                        logger.debug(f"mask padded to match key length: {tuple(selected_mask.shape)}")

            return selected_mask
        else:
            if logger:
                shapes_info = []
                for i, t in enumerate(attention_mask):
                    if torch.is_tensor(t):
                        shapes_info.append(f"[{i}]: {tuple(t.shape)}")
                    else:
                        shapes_info.append(f"[{i}]: {type(t)}")
                logger.warning(f"No suitable mask found in attention_mask tuple. Need ({Lq}, {Lk}) but got: {', '.join(shapes_info)}. Falling back to None")
            return None

    # Any other type is silently ignored
    if logger:
        logger.warning(f"Unsupported attention_mask type: {type(attention_mask)}, falling back to None")
    return None


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16, network_alpha=None, device=None, dtype=None):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class FluxAttnProcessor2_0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None, camera_embeds=None, perform_swap=False, batch_swap_ids=None, is_i2i=False, **kwargs):
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Shape-aware mask selection (compatible with the Diffusers 0.35+ tuple format)
        selected_mask = _pick_sdpa_mask(attention_mask, query, key, attn.heads)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=selected_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class BokehFluxAttnProcessor2_0(torch.nn.Module):
    """
    Bokeh-aware Flux Attention Processor with Grounded Attention support.

    Supports two Grounded Attention modes:
    1. T2I mode: a randomly chosen sample in the batch is used as the pivot.
    2. I2I mode: the input image (batch[0]) is used as the pivot to lock down
       the scene structure.

    Grounded Attention implements the paper's formulation:
    Attention(Q_piv, [K_tgt, K_piv], [V_tgt, V_tgt])

    where:
    - Q_piv: query from the pivot sample (shared by every sample)
    - K_tgt, K_piv: concatenated keys from the current sample and the pivot
    - V_tgt, V_tgt: two copies of the current sample's value
    """
    def __init__(self, context_dim, hidden_dim, block_name=None, bokeh_scale=1.0,
                 unfreeze_q=False, unfreeze_k=False, attn_bias=False, lora_rank=None, lora_scale=1.0, lora_alpha=None, logger=None):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.block_name = block_name
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.bokeh_scale = bokeh_scale
        self.lora_scale = lora_scale
        self.unfreeze_q = unfreeze_q
        self.unfreeze_k = unfreeze_k
        self.logger = logger or logging.getLogger(__name__)

        if lora_rank is not None:
            self.to_k_dof = LoRALinear(context_dim, hidden_dim,
                                       rank=lora_rank, network_alpha=lora_alpha)
            self.to_v_dof = LoRALinear(context_dim, hidden_dim,
                                       rank=lora_rank, network_alpha=lora_alpha)
            if self.unfreeze_q:
                self.to_q_adp = LoRALinear(hidden_dim, hidden_dim,
                                           rank=lora_rank, network_alpha=lora_alpha)
            if self.unfreeze_k:
                self.to_k_adp = LoRALinear(hidden_dim, hidden_dim,
                                           rank=lora_rank, network_alpha=lora_alpha)
        else:
            self.to_k_dof = nn.Linear(context_dim, hidden_dim, bias=attn_bias)
            self.to_v_dof = nn.Linear(context_dim, hidden_dim, bias=attn_bias)
            nn.init.zeros_(self.to_k_dof.weight)
            nn.init.zeros_(self.to_v_dof.weight)
            if attn_bias:
                nn.init.zeros_(self.to_k_dof.bias)
                nn.init.zeros_(self.to_v_dof.bias)
            if self.unfreeze_q:
                self.to_q_adp = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)
                nn.init.zeros_(self.to_q_adp.weight)
                if attn_bias:
                    nn.init.zeros_(self.to_q_adp.bias)
            if self.unfreeze_k:
                self.to_k_adp = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)
                nn.init.zeros_(self.to_k_adp.weight)
                if attn_bias:
                    nn.init.zeros_(self.to_k_adp.bias)

    def _resolve_runtime_args(self, camera_embeds, perform_swap, batch_swap_ids, is_i2i):
        """
        Resolve runtime arguments: explicitly passed values win, otherwise fall
        back to stored attributes.
        Note: perform_swap / is_i2i are controlled solely by
        ``enable_grounded_attention`` -- whatever the caller passes in is used.
        """

        # Avoid creating a duplicate logger to prevent duplicate log output.
        # logger = logging.getLogger(__name__)

        # Handle camera_embeds
        if camera_embeds is None and hasattr(self, 'stored_camera_embeds') and hasattr(self, 'use_stored_embeds') and self.use_stored_embeds:
            camera_embeds = self.stored_camera_embeds

        # Handle grounded-attention args -- only controlled by external input.
        if not perform_swap and hasattr(self, 'stored_perform_swap'):
            perform_swap = self.stored_perform_swap
            batch_swap_ids = self.stored_batch_swap_ids
            is_i2i = self.stored_is_i2i

            # Debug: show which source provided the values.
            #if perform_swap:
                #logger.debug(f"Block {self.block_name}: using stored grounded-attention args: perform_swap={perform_swap}, is_i2i={is_i2i}, batch_swap_ids={batch_swap_ids}")

        return camera_embeds, perform_swap, batch_swap_ids, is_i2i

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None, camera_embeds=None, perform_swap=False, batch_swap_ids=None, is_i2i=False, **kwargs):
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # Args are already declared in the function signature; no need to pull them from kwargs.

        # Resolve runtime arguments.
        camera_embeds, perform_swap, batch_swap_ids, is_i2i = self._resolve_runtime_args(
            camera_embeds, perform_swap, batch_swap_ids, is_i2i
        )

        # Debug: print basic info at the start of the method.
        #print(f"[DEBUG] BokehFluxAttnProcessor2_0.__call__ - block_name: {self.block_name}")
        #print(f"[DEBUG] hidden_states.shape: {hidden_states.shape}")
        #print(f"[DEBUG] perform_swap: {perform_swap}, is_i2i: {is_i2i}")

        # `sample` projections.
        query = attn.to_q(hidden_states)  # [batch_size, seq_len, hidden_dim]
        key = attn.to_k(hidden_states)  # [batch_size, seq_len, hidden_dim]
        value = attn.to_v(hidden_states)  # [batch_size, seq_len, hidden_dim]

        if self.unfreeze_q:
            query = query + self.lora_scale * self.to_q_adp(query)
        if self.unfreeze_k:
            key = key + self.lora_scale * self.to_k_adp(key)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Grounded Attention implementation
        if perform_swap:
            # Avoid creating a duplicate logger to prevent duplicate log output.
            # logger = logging.getLogger(__name__)

            if is_i2i:
                # I2I mode: the pivot is the input image (fixed as batch[0], external batch_swap_ids ignored).
                # Validate batch size and structure.
                if batch_size < 1:
                    # logger.warning(f"Block {self.block_name}: I2I mode but batch_size={batch_size}, cannot apply grounded attention")
                    perform_swap = False
                else:
                    pivot_idx = 0  # I2I scenes always use the first image as the pivot
                    query_pivot = query[pivot_idx:pivot_idx+1]  # query of the input image
                    # Replace every sample's query with the input image's query (shared query).
                    # Use repeat + contiguous instead of expand_as for better numerical stability.
                    query = query_pivot.repeat(batch_size, 1, 1, 1).contiguous()

                    #logger.debug(f"Block {self.block_name}: I2I mode - using input image as pivot (idx={pivot_idx}), ignoring batch_swap_ids")
                    #logger.debug(f"Block {self.block_name}: I2I mode - batch_size={batch_size}, query.shape after sharing: {query.shape}")

                    # I2I mode: concatenate the input image's key with the current key.
                    key_pivot = key[pivot_idx:pivot_idx+1]  # key of the input image
                    # Expand the pivot key over the whole batch (repeat + contiguous).
                    key_pivot_expanded = key_pivot.repeat(batch_size, 1, 1, 1).contiguous()
                    key = torch.cat([key, key_pivot_expanded], dim=2)  # [Ktgt, Kpiv]
                    value = torch.cat([value, value], dim=2)  # [Vtgt, Vtgt] - duplicate current value

                    #logger.debug(f"Block {self.block_name}: I2I mode - key/value concatenation finished (no unfreeze_k restriction)")
                    #logger.debug(f"Block {self.block_name}: I2I mode - key.shape: {key.shape}, value.shape: {value.shape}")
                    #logger.debug(f"Block {self.block_name}: I2I mode - implements Attention(Q_piv, [K_tgt,K_piv], [V_tgt,V_tgt])")
            else:
                # T2I mode: aligned with bokeh_adapter_flux_T2I.py; only acts during self-attention.
                if encoder_hidden_states is None and batch_swap_ids is not None:
                    # Only swap queries when batch_swap_ids is explicitly provided.
                    query = query[batch_swap_ids]
                    # Only concatenate K/V when unfreeze_k=True.
                    if self.unfreeze_k:
                        key_swaps = key[batch_swap_ids]
                        key = torch.cat([key, key_swaps], dim=2)
                        value = torch.cat([value, value], dim=2)

        # Shape-aware mask selection (compatible with the Diffusers 0.35+ tuple format)
        selected_mask = _pick_sdpa_mask(attention_mask, query, key, attn.heads, self.logger)
        # Warn if the picked mask is None but the original mask was not.
        if selected_mask is None and attention_mask is not None:
            self.logger.warning(f"Block {self.block_name}: attention mask reset to None (may degrade attention quality)")

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=selected_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if camera_embeds is not None:
            # Debug info
            #print(f"[DEBUG] camera_embeds is in use: shape={camera_embeds.shape}, mean={camera_embeds.mean():.6f}")

            key_dof = self.to_k_dof(camera_embeds)
            value_dof = self.to_v_dof(camera_embeds)
            key_dof = key_dof.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_dof = value_dof.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # Camera attention always uses None as the mask (does not involve the user-supplied attention_mask).
            camera_hidden_states = F.scaled_dot_product_attention(query, key_dof, value_dof, attn_mask=None, dropout_p=0.0, is_causal=False)
            with torch.no_grad():
                self.attn_map = query @ key_dof.transpose(-2, -1)
            camera_hidden_states = camera_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            camera_hidden_states = camera_hidden_states.to(query.dtype)
            hidden_states = hidden_states + self.bokeh_scale * camera_hidden_states

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class BokehFluxControlAdapter(torch.nn.Module):
    def __init__(
        self,
        base_model,
        blocks,
        ckpt_path=None,
        bokeh_scale=1.0,
        context_dim=768,
        hidden_dim=3072,
        lora_rank=None,
        lora_alpha=None,
        lora_scale=1.0,
        unfreeze_q=False,
        unfreeze_k=False,
        logger=None,
        mode: str = "i2i",  # interface mode, accepts 't2i' or 'i2i'
    ):
        super().__init__()
        self.blocks = blocks
        self.bokeh_scale = bokeh_scale
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scale = lora_scale
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.unfreeze_q = unfreeze_q
        self.unfreeze_k = unfreeze_k
        self.logger = logger or logging.getLogger(__name__)
        self.mode = mode.lower() if isinstance(mode, str) else "i2i"

        # Input dim is 1; only the bokeh strength K is fed in.
        self.embedding_layer = get_simple_embedding_layer(1, self.context_dim)
        self._attach_transformer(base_model, blocks)

        if ckpt_path is not None:
            self._load_from_pretrained_checkpoint(ckpt_path)

        self.cross_attn_maps = {}
        self._register_cross_attention_hook(base_model, blocks)

    def _attach_transformer(self, base_model, blocks):
        # T2I mode: keep the original processor on un-replaced blocks; only the
        # adapters we create are collected into ``adapter_modules``.
        # I2I mode: keep the existing behavior (un-replaced blocks use the
        # default FluxAttnProcessor2_0; ``adapter_modules`` collects all of them).
        attn_procs = {}
        created_adapters = []
        replaced_count = 0

        # Read current processors (may be an attribute or a callable returning a dict).
        current_attn_procs = None
        if hasattr(base_model, 'attn_processors'):
            current_attn_procs = base_model.attn_processors() if callable(base_model.attn_processors) else base_model.attn_processors

        for name in (current_attn_procs.keys() if current_attn_procs is not None else base_model.attn_processors.keys()):
            block_name = name.split('.attn')[0]
            if block_name in blocks:
                proc = BokehFluxAttnProcessor2_0(
                    context_dim=self.context_dim,
                    hidden_dim=self.hidden_dim,
                    block_name=block_name,
                    attn_bias=False,
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    unfreeze_q=self.unfreeze_q,
                    unfreeze_k=self.unfreeze_k,
                    logger=self.logger,
                )
                attn_procs[name] = proc
                created_adapters.append(proc)
                replaced_count += 1
            else:
                if self.mode == 't2i' and current_attn_procs is not None:
                    # T2I mode: keep the original processor.
                    attn_procs[name] = current_attn_procs[name]
                else:
                    # I2I mode: use the default FluxAttnProcessor2_0.
                    attn_procs[name] = FluxAttnProcessor2_0()

        base_model.set_attn_processor(attn_procs)

        if self.mode == 't2i':
            # Only collect the adapters we created, matching the T2I variant.
            self.adapter_modules = torch.nn.ModuleList(created_adapters)
        else:
            # Keep the original behavior: collect every processor.
            self.adapter_modules = torch.nn.ModuleList(base_model.attn_processors.values())

        # Debug: print how many layers were replaced.
        #print(f"[DEBUG] total attention layers: {len(base_model.attn_processors)}")
        #print(f"[DEBUG] replaced layers: {replaced_count}")
        #print(f"[DEBUG] replaced layer names: {[name for name in base_model.attn_processors.keys() if name.split('.attn')[0] in blocks]}")

    def _load_from_pretrained_checkpoint(self, ckpt_path: str):
        """
        Load weights from a pretrained checkpoint while skipping the
        incompatible ``embedding_layer.0.weight``. Designed for the legacy
        adapter-only pretrained bins: the model went 1D -> 2D -> 1D input
        across revisions, and even though we are back to 1D the historical
        weights still need to be skipped when their shape no longer matches.
        """
        # Allow passing a directory; auto-locate pytorch_model.bin or model.safetensors.
        if os.path.isdir(ckpt_path):
            # Prefer the safetensors format.
            safetensors_candidate = os.path.join(ckpt_path, "model.safetensors")
            pytorch_candidate = os.path.join(ckpt_path, "pytorch_model.bin")

            if os.path.exists(safetensors_candidate):
                ckpt_path = safetensors_candidate
            elif os.path.exists(pytorch_candidate):
                ckpt_path = pytorch_candidate
            else:
                raise FileNotFoundError(f"Neither model.safetensors nor pytorch_model.bin found in directory: {ckpt_path}")

        orig_emb_sum = torch.sum(torch.stack([torch.sum(p) for p in self.embedding_layer.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Pick the loader based on the file extension.
        if ckpt_path.endswith('.safetensors'):
            print(f"[load] reading safetensors: {ckpt_path}")
            state_dict = load_safetensors(ckpt_path)
        else:
            try:
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            except (TypeError, Exception) as e:
                print(f"[load][warn] weights_only=True failed, retrying with weights_only=False: {e}")
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # embedding_layer: load 0.weight only when the shape matches (no blanket skip).
        embedding_dict = {}
        skipped_keys = []
        current_first_weight_shape = tuple(self.embedding_layer[0].weight.shape)
        for k, v in state_dict.items():
            if k.startswith("embedding_layer."):
                key_name = k.replace("embedding_layer.", "")
                if key_name == "0.weight":
                    if tuple(v.shape) != current_first_weight_shape:
                        skipped_keys.append(k)
                        print(f"[load][warn] skipping incompatible weight: {k} {list(v.shape)} -> expected shape {list(current_first_weight_shape)}")
                        continue
                embedding_dict[key_name] = v

        # adapter_modules: prefer prefixed keys; fall back to numeric-index keys (ModuleList style).
        adapter_dict = {k.replace("adapter_modules.", ""): v for k, v in state_dict.items() if k.startswith("adapter_modules.")}
        adapter_numeric_dict = {k: v for k, v in state_dict.items() if k.split(".")[0].isdigit()}

        # Load the embedding layer (strict).
        if embedding_dict:
            self.embedding_layer.load_state_dict(embedding_dict, strict=True)

        # Load the adapter modules (prefix first, then numeric indices); strict=False for robustness.
        loaded_adapter = False
        if adapter_dict:
            try:
                self.adapter_modules.load_state_dict(adapter_dict, strict=True)
                loaded_adapter = True
            except Exception as e:
                print(f"[load][warn] prefix-based adapter load failed, retrying with numeric-index keys (strict): {e}")
        if (not loaded_adapter) and adapter_numeric_dict:
            self.adapter_modules.load_state_dict(adapter_numeric_dict, strict=True)

        new_emb_sum = torch.sum(torch.stack([torch.sum(p) for p in self.embedding_layer.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Sanity-check that the adapter weights actually changed (only when adapter weights were provided).
        if adapter_dict or adapter_numeric_dict:
            assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"[load] loaded from pretrained checkpoint: {ckpt_path}")
        print(f"[load] summary:")
        print(f"   - skipped weights: {len(skipped_keys)}")
        print(f"   - embedding weights (excluding 0.weight): {len(embedding_dict)} loaded")
        loaded_adapters_num = len(adapter_dict) if adapter_dict else len(adapter_numeric_dict)
        print(f"   - adapter weights: {loaded_adapters_num} loaded")

    def _clear_attn_maps(self):
        self.cross_attn_maps.clear()

    def _hook_fn(self, name):
        def forward_hook(module, input, output):
            if hasattr(module.processor, "attn_map"):
                self.cross_attn_maps[name] = module.processor.attn_map
                del module.processor.attn_map
        return forward_hook

    def _register_cross_attention_hook(self, base_model, blocks):
        for name, module in base_model.named_modules():
            block_name = name.split(".attn")[0]
            if name.split('.')[-1].startswith('attn') and block_name in blocks:
                module.register_forward_hook(self._hook_fn(name))
        return base_model

    def set_lora_scale(self, lora_scale):
        self.lora_scale = lora_scale
        for module in self.adapter_modules:
            module.lora_scale = lora_scale

    def set_bokeh_scale(self, bokeh_scale):
        self.bokeh_scale = bokeh_scale
        for module in self.adapter_modules:
            module.bokeh_scale = bokeh_scale

    def forward(self, base_model, camera_ann, hidden_states, timestep, guidance, pooled_projections,
                encoder_hidden_states, txt_ids, img_ids, perform_swap=False, batch_swap_ids=None, is_i2i=None):
        """
        Forward pass with enhanced camera parameter control.

        Args:
            camera_ann: Tensor of shape [batch_size, 1] containing [K] parameters
                       - K: bokeh strength (normalized 0-1, maps to 0.0-30.0 range)
            perform_swap: Enable grounded attention mechanism
            batch_swap_ids: Indices for attention swapping (T2I mode)
            is_i2i: Whether this is image-to-image mode
        """
        camera_embeds = self.embedding_layer(camera_ann)

        # Decide whether we are running I2I based on the adapter mode (when the caller did not specify).
        if is_i2i is None:
            is_i2i = (self.mode == 'i2i')
        noise_pred = base_model(
            hidden_states,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_projections,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids,
            joint_attention_kwargs={
                "camera_embeds": camera_embeds,
                "perform_swap": perform_swap,
                "batch_swap_ids": batch_swap_ids,
                "is_i2i": is_i2i,  # mode flag: I2I or T2I
            },
            return_dict=False,
        )[0]
        return noise_pred

    def get_net_attn_map(self, batch_size=1, idx=0, detach=True):
        net_attn_maps = {}
        for name, attn_map in self.cross_attn_maps.items():
            attn_map = attn_map.cpu() if detach else attn_map
            attn_map = torch.chunk(attn_map, batch_size)[idx].squeeze()

            side_length = int(np.sqrt(attn_map.shape[1]))
            attn_map = attn_map.view(-1, side_length, side_length)
            attn_map = torch.mean(attn_map, dim=0)

            net_attn_maps[name] = attn_map
        self._clear_attn_maps()
        return net_attn_maps
