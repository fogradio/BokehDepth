#!/usr/bin/env python3
"""Run Bokeh Diffusion stage-1 + UniDepthV2-DSFA stage-2 on a single image."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[0]
BOKEH_DEV_DIR = REPO_ROOT / "bokeh-generation"
UNIDEPTH_DIR = REPO_ROOT / "UniDepth"

for extra_path in (BOKEH_DEV_DIR, UNIDEPTH_DIR):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

# Stage-1 imports (bokeh diffusion)
from gen_bokeh_stack import (  # type: ignore  # noqa: E402
    build_prompt_from_template,
    generate_i2i_bokeh_image,
    parse_block_ids,
)
from model.utils import color_transfer_lab  # type: ignore  # noqa: E402
from model.bokeh_adapter_flux import BokehFluxControlAdapter  # type: ignore  # noqa: E402
from diffusers import FluxKontextPipeline  # type: ignore  # noqa: E402
from constants import FLUX_TRANSFORMER_BLOCKS  # type: ignore  # noqa: E402

# Stage-2 imports (UniDepth DSFA)
from unidepth.utils.camera import BatchCamera, Pinhole  # type: ignore  # noqa: E402
from unidepth.utils.constants import (  # type: ignore  # noqa: E402
    IMAGENET_DATASET_MEAN,
    IMAGENET_DATASET_STD,
)
from unidepth.utils.visualization import colorize  # type: ignore  # noqa: E402
from unidepth.models.unidepthv2.unidepthv2_DSFA import (  # type: ignore  # noqa: E402
    _postprocess,
    _postprocess_intrinsics,
    get_paddings,
    get_resize_factor,
)

from unidepth.checkpoint import (  # type: ignore  # noqa: E402
    instantiate_model as unidepth_instantiate_model,
    load_config as unidepth_load_config,
)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


def _now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class Stage1Artifacts:
    ref_image_path: Path
    stack_dir: Path
    stack_paths: List[Path]
    k_values: List[float]
    stack_index_path: Path
    size: Tuple[int, int]
    source_size: Tuple[int, int]


# ---------------------------------------------------------------------------
# Stage-1: Bokeh stack generator
# ---------------------------------------------------------------------------


def _select_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _load_bokeh_components(
    pretrained_model_name: str,
    adapter_ckpt: Path,
    blocks: Sequence[int],
    mixed_precision: str,
    lora_rank: int,
    lora_alpha: float,
    unfreeze_q: bool,
    unfreeze_k: bool,
    device: torch.device,
):
    adapter_ckpt = Path(adapter_ckpt)
    if not adapter_ckpt.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_ckpt}")

    weight_dtype = _select_dtype(mixed_precision)
    pipeline = FluxKontextPipeline.from_pretrained(
        pretrained_model_name,
        torch_dtype=weight_dtype,
    )
    transformer = pipeline.transformer
    vae = pipeline.vae
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    pipeline = pipeline.to(device, dtype=weight_dtype)
    adapter = BokehFluxControlAdapter(
        transformer,
        blocks=blocks,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        unfreeze_q=unfreeze_q,
        unfreeze_k=unfreeze_k,
        ckpt_path=str(adapter_ckpt),
    ).to(device, dtype=weight_dtype)

    if not any(p.requires_grad for p in adapter.parameters()):
        raise RuntimeError(
            "[Stage1] Adapter has no parameters; check block_ids/FLUX_TRANSFORMER_BLOCKS mapping."
        )
    adapter.eval()
    vae_scale_factor = getattr(pipeline, "vae_scale_factor", getattr(vae, "scale_factor", 8))
    return pipeline, adapter, weight_dtype, vae_scale_factor


def _resize_short_side(img: Image.Image, short_side: int | None) -> Image.Image:
    if short_side is None:
        return img
    short_side = max(1, int(short_side))
    w, h = img.size
    current_short = min(w, h)
    if current_short == short_side:
        return img
    scale = short_side / current_short
    new_w = max(8, int(round(w * scale)))
    new_h = max(8, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.LANCZOS)


def run_stage1(
    args,
    out_dir: Path,
    device: torch.device,
) -> Stage1Artifacts:
    out_dir.mkdir(parents=True, exist_ok=True)
    stack_dir = out_dir / "defocus_stack"
    stack_dir.mkdir(parents=True, exist_ok=True)

    ref_img = Image.open(args.ref_image).convert("RGB")
    orig_w, orig_h = ref_img.size

    target_w = max(1, int(args.ref_width)) if args.ref_width else None
    target_h = max(1, int(args.ref_height)) if args.ref_height else None

    ref_base = ref_img
    if target_w is not None and target_h is not None:
        ref_base = ref_img.resize((target_w, target_h), Image.LANCZOS)

    source_w, source_h = ref_base.size
    if args.short_side is not None and args.short_side > 0:
        ref_resized = _resize_short_side(ref_base, args.short_side)
    else:
        ref_resized = ref_base

    ref_copy_path = out_dir / "ref.png"
    ref_resized.save(ref_copy_path)

    raw_block_ids = parse_block_ids(args.block_ids)
    blocks = [FLUX_TRANSFORMER_BLOCKS[int(i)] for i in raw_block_ids]
    pipeline, adapter, weight_dtype, vae_scale = _load_bokeh_components(
        pretrained_model_name=args.pretrained_model,
        adapter_ckpt=Path(args.adapter_ckpt),
        blocks=blocks,
        mixed_precision=args.mixed_precision,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        unfreeze_q=args.unfreeze_q,
        unfreeze_k=args.unfreeze_k,
        device=device,
    )
    print(
        f"[Stage1] Loaded Flux + adapter weights "
        f"from {args.pretrained_model} / {args.adapter_ckpt}"
    )

    prompts = [build_prompt_from_template(args.prompt_template, k) for k in args.k_values]
    with torch.inference_mode():
        generated = generate_i2i_bokeh_image(
            pipeline,
            adapter,
            input_image=ref_resized,
            prompt=prompts,
            dof_cond=None,
            k_values_batch=args.k_values,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            device=device,
            seed=args.seed,
            dtype=weight_dtype,
            vae_scale_factor=vae_scale,
            assigned_size=None,
        )
    if isinstance(generated, Image.Image):
        generated_images = [generated]
    else:
        generated_images = list(generated)

    if args.apply_color_transfer:
        ct_images = []
        for img in generated_images:
            ct_img = color_transfer_lab(ref_resized, img)
            ct_images.append(ct_img)
        generated_images = ct_images

    stack_paths: List[Path] = []
    k_map: dict[str, float] = {}
    for idx, (img, k) in enumerate(zip(generated_images, args.k_values)):
        fname = f"{idx}.png"
        img_path = stack_dir / fname
        img.save(img_path)
        stack_paths.append(img_path)
        k_map[fname.split(".")[0]] = float(k)

    stack_index = {
        "ids": [str(i) for i in range(len(stack_paths))],
        "k_values": {str(i): float(k) for i, k in enumerate(args.k_values)},
        "ref_image": str(ref_copy_path),
        "source_rgb": str(ref_copy_path),
        "size": [ref_resized.width, ref_resized.height],
        "source_size": [source_w, source_h],
        "original_size": [orig_w, orig_h],
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_steps,
        "prompt_template": args.prompt_template,
        "method": "bokeh_diffusion_i2i_pipeline",
    }
    stack_index_path = stack_dir / "stack_index.json"
    stack_index_path.write_text(json.dumps(stack_index, indent=2), encoding="utf-8")

    manifest_entry = {
        "dataset": "bokehdepth",
        "timestamp": _now_timestamp(),
        "ref": str(ref_copy_path),
        "stack": [str(p) for p in stack_paths],
        "k": [float(k) for k in args.k_values],
        "stack_index": str(stack_index_path),
        "size": [ref_resized.width, ref_resized.height],
        "source_size": [source_w, source_h],
    }
    (out_dir / "manifest.jsonl").write_text(
        json.dumps(manifest_entry, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Release stage-1 models to free VRAM
    del pipeline
    del adapter
    torch.cuda.empty_cache()

    return Stage1Artifacts(
        ref_image_path=ref_copy_path,
        stack_dir=stack_dir,
        stack_paths=stack_paths,
        k_values=[float(k) for k in args.k_values],
        stack_index_path=stack_index_path,
        size=(ref_resized.height, ref_resized.width),
        source_size=(source_h, source_w),
    )


def _build_default_camera(width: int, height: int, device: torch.device):
    fx = fy = 0.7 * width
    cx = width * 0.5
    cy = height * 0.5
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32
    ).unsqueeze(0)
    camera = BatchCamera.from_camera(Pinhole(K=K))
    return camera.to(device)


def _tensor_from_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = torch.from_numpy(np.asarray(img, dtype=np.uint8))
    return arr.permute(2, 0, 1).contiguous()


def _prepare_stack_tensors(stack_paths: Sequence[Path]) -> torch.Tensor:
    tensors = [_tensor_from_image(p).float() for p in stack_paths]
    return torch.stack(tensors, dim=0)


def _normalize_img(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    tensor = tensor / 255.0
    return (tensor - mean) / std


def _build_stage2_context(config: dict, device: torch.device):
    training_cfg = config.get("training", {})
    use_fp16 = bool(training_cfg.get("f16", False)) and device.type == "cuda"
    if not use_fp16:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def run_stage2(
    args,
    artifacts: Stage1Artifacts,
    stage_dir: Path,
    device: torch.device,
):
    config = unidepth_load_config(Path(args.config))
    model = unidepth_instantiate_model(config, Path(args.weights), device=device)
    print(f"[Stage2] Loaded UniDepth weights from {args.weights}")
    if args.resolution_level is not None:
        level = int(args.resolution_level)
        setattr(model, "resolution_level", level)

    rgb_tensor = _tensor_from_image(artifacts.ref_image_path).float()
    stack_tensor = _prepare_stack_tensors(artifacts.stack_paths)

    H, W = rgb_tensor.shape[-2:]
    ratio_bounds = model.shape_constraints["ratio_bounds"]
    pixels_bounds = [
        model.shape_constraints["pixels_min"],
        model.shape_constraints["pixels_max"],
    ]
    paddings, padded_hw = get_paddings((H, W), ratio_bounds)
    resize_factor, resized_hw = get_resize_factor(padded_hw, pixels_bounds)

    pad_left, pad_right, pad_top, pad_bottom = paddings
    pad_tuple = (pad_left, pad_right, pad_top, pad_bottom)

    mean = torch.tensor(IMAGENET_DATASET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DATASET_STD, device=device).view(1, 3, 1, 1)

    rgb = rgb_tensor.unsqueeze(0).to(device)
    rgb = F.pad(rgb, pad_tuple, value=0.0)
    rgb = F.interpolate(rgb, size=resized_hw, mode="bilinear", align_corners=False)
    rgb = _normalize_img(rgb, mean, std)

    stack = stack_tensor.to(device)
    stack = F.pad(stack, pad_tuple, value=0.0)
    stack = F.interpolate(stack, size=resized_hw, mode="bilinear", align_corners=False)
    stack = _normalize_img(stack, mean, std)
    stack = stack.unsqueeze(0)  # [B=1, S, C, H, W]

    focus_k = torch.tensor(artifacts.k_values, dtype=torch.float32, device=device)
    focus_k = focus_k.unsqueeze(0)

    camera = _build_default_camera(W, H, device)
    camera = camera.crop(
        left=-pad_left,
        top=-pad_top,
        right=-pad_right,
        bottom=-pad_bottom,
    )
    camera = camera.resize(resize_factor)

    inputs = {
        "image": rgb,
        "camera": camera,
        "defocus_stack": stack,
        "defocus_k": focus_k,
    }

    context = _build_stage2_context(config, device)

    with torch.inference_mode(), context:
        _, outputs = model.encode_decode(inputs, image_metas=[])

    depth = outputs["depth"]
    depth = _postprocess(depth, (H, W), paddings).squeeze(0).squeeze(0)
    depth_np = depth.detach().cpu().numpy().astype(np.float32)

    vis = _render_depth(depth_np)
    depth_path = stage_dir / "depth.npy"
    color_path = stage_dir / "depth_color.png"
    np.save(depth_path, depth_np)
    Image.fromarray(vis).save(color_path)

    meta = {
        "ref_image": str(artifacts.ref_image_path),
        "stack_index": str(artifacts.stack_index_path),
        "k_values": artifacts.k_values,
        "config": str(args.config),
        "weights": str(args.weights),
        "resolution_level": args.resolution_level,
        "depth_path": str(depth_path),
        "depth_color": str(color_path),
    }
    (stage_dir / "stage2_summary.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    return depth_path, color_path


def _render_depth(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth) & (depth > 0)
    if not np.any(finite):
        vmin, vmax = float(depth.min()), float(depth.max())
    else:
        values = depth[finite]
        vmin = float(np.percentile(values, 5))
        vmax = float(np.percentile(values, 95))
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-3
    colored = colorize(depth, vmin=vmin, vmax=vmax, cmap="magma_r")
    return colored


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a bokeh defocus stack with bokeh-diffusion and run UniDepthV2 "
            "DSFA depth estimation on the result."
        )
    )

    # General
    parser.add_argument("--ref-image", required=True, type=Path, help="Input RGB image path")
    parser.add_argument(
        "--ref-width",
        type=int,
        default=512,
        help="Resize reference image to this width before further processing",
    )
    parser.add_argument(
        "--ref-height",
        type=int,
        default=512,
        help="Resize reference image to this height before further processing",
    )
    parser.add_argument(
        "--k-values",
        type=float,
        nargs="+",
        required=True,
        help="List of aperture strength values (e.g. 10 20 30)",
    )
    parser.add_argument(
        "--short-side",
        type=int,
        default=None,
        help="Optional short-side resolution to resize the reference image before stage-1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "examples",
        help="Directory where per-run timestamped folders will be created",
    )
    parser.add_argument("--seed", type=int, default=42)

    # Stage-1 options
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="black-forest-labs/FLUX.1-Kontext-dev",
        help="HuggingFace ID or local path for Flux Kontext base model",
    )
    parser.add_argument(
        "--adapter-ckpt",
        type=Path,
        default=REPO_ROOT / "weights" / "bokeh_lora.bin",
        help="Bokeh adapter checkpoint path",
    )
    parser.add_argument(
        "--block-ids",
        type=str,
        default="0-56",
        help="Adapter block ids (same syntax as stage-1 script, e.g. '0-56' or '0,8,16')",
    )
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--lora-alpha", type=float, default=128.0)
    parser.add_argument("--unfreeze-q", action="store_true", help="Unfreeze transformer Q projections")
    parser.add_argument("--unfreeze-k", action="store_true", help="Unfreeze transformer K projections")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=(
            "Set dof_cond = {value:.2f} (stronger background defocus); "
            "preserve subject sharpness; keep composition, lighting, and colors unchanged."
        ),
        help="Format string template used to build prompts for each K value",
    )
    parser.add_argument(
        "--apply-color-transfer",
        action="store_true",
        help="Apply LAB color transfer from reference image to generated stack",
    )

    # Stage-2 options
    parser.add_argument(
        "--config",
        type=Path,
        default=UNIDEPTH_DIR / "configs" / "config_v2_vitl14_DSFA_inference.json",
        help="UniDepth model config JSON",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "weights" / "UDv2_dsfa_release.pth",
        help="UniDepth checkpoint to load",
    )
    parser.add_argument(
        "--resolution-level",
        type=int,
        default=None,
        help="Optional resolution level override (0-9) matching eval scripts",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for both stages (default: cuda if available)",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA device requested but torch.cuda.is_available() is False")

    run_dir = args.output_root / f"run_{_now_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing outputs to {run_dir}")

    artifacts = run_stage1(args, run_dir, device)
    depth_path, color_path = run_stage2(args, artifacts, run_dir, device)

    summary = {
        "ref_image": str(args.ref_image),
        "stage1": {
            "adapter": str(args.adapter_ckpt),
            "pretrained_model": args.pretrained_model,
            "k_values": artifacts.k_values,
            "stack_dir": str(artifacts.stack_dir),
        },
        "stage2": {
            "config": str(args.config),
            "weights": str(args.weights),
            "depth_path": str(depth_path),
            "depth_color": str(color_path),
        },
    }
    (run_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("[INFO] Pipeline finished successfully")
    print(f"       Depth map : {depth_path}")
    print(f"       Colored   : {color_path}")


if __name__ == "__main__":
    main()
