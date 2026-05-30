# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import torch
import torch.nn.functional as F

from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.use_sdpa = use_sdpa

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape

        # Priority 1: attention bias requires xFormers.
        if attn_bias is not None:
            if not XFORMERS_AVAILABLE:
                raise RuntimeError("attn_bias requires xFormers but it's not available")

            # Keep qkv projection weights aligned with the input dtype.
            if self.qkv.weight.dtype != x.dtype:
                self.qkv = self.qkv.to(x.dtype)
                self.proj = self.proj.to(x.dtype)

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = unbind(qkv, 2)
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
            x = x.reshape([B, N, C])
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        # Priority 2: use PyTorch SDPA when available.
        if self.use_sdpa and x.is_cuda:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

            # Official PyTorch SDPA path.
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
            )
            x = out.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        # Priority 3: use xFormers when available.
        can_use_xformers = (
            XFORMERS_AVAILABLE and
            x.device.type == 'cuda' and
            x.dtype in [torch.float16, torch.bfloat16]
        )

        if can_use_xformers:
            # Keep qkv projection weights aligned with the input dtype.
            if self.qkv.weight.dtype != x.dtype:
                self.qkv = self.qkv.to(x.dtype)
                self.proj = self.proj.to(x.dtype)

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = unbind(qkv, 2)
            x = memory_efficient_attention(q, k, v, attn_bias=None)
            x = x.reshape([B, N, C])
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        # Priority 4: fall back to the plain attention implementation.
        return super().forward(x)
