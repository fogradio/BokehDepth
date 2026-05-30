#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F


matplotlib.use("Agg")

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from depth_anything_v2 import MODEL_CONFIGS, DepthAnythingV2DSFA  # noqa: E402


def load_state(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    return {
        key.removeprefix("module."): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def read_sample(path: Path) -> tuple[Path, list[Path], list[float]]:
    if path.suffix == ".jsonl":
        line = next(
            item.strip()
            for item in path.read_text(encoding="utf-8").splitlines()
            if item.strip()
        )
        sample = json.loads(line)
    else:
        sample = json.loads(path.read_text(encoding="utf-8"))

    base_dir = path.parent
    ref_value = sample.get("ref") or sample.get("source_rgb") or sample.get("image_path")
    if not ref_value:
        raise ValueError("Sample must contain ref, source_rgb, or image_path")

    if "stack" in sample and "k" in sample:
        stack_values = sample["stack"]
        k_values = sample["k"]
        stack_paths = [resolve_path(item, base_dir) for item in stack_values]
    elif "ids" in sample and "k_values" in sample:
        stack_index_dir = base_dir
        if "stack_index" in sample:
            stack_index_dir = resolve_path(sample["stack_index"], base_dir).parent
        elif "stack_dir" in sample:
            stack_index_dir = resolve_path(sample["stack_dir"], base_dir)

        ids = [str(item) for item in sample["ids"]]
        k_raw = sample["k_values"]
        if isinstance(k_raw, dict):
            k_values = [k_raw[item] for item in ids]
        else:
            k_values = k_raw
        stack_paths = [stack_index_dir / f"{item}.png" for item in ids]
    else:
        raise ValueError("Sample must contain stack/k or ids/k_values")

    return resolve_path(ref_value, base_dir), stack_paths, [float(item) for item in k_values]


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (base_dir / resolved).resolve()


def select_stack(
    stack_paths: Sequence[Path],
    k_values: Sequence[float],
    indices: Sequence[int] | None,
) -> tuple[list[Path], list[float]]:
    if indices is None:
        return list(stack_paths), [float(item) for item in k_values]
    selected = [idx for idx in indices if 0 <= idx < len(stack_paths)]
    if not selected:
        raise ValueError("stack-indices did not select any valid frame")
    return [stack_paths[idx] for idx in selected], [float(k_values[idx]) for idx in selected]


def render_depth(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth) & (depth > 0)
    if np.any(finite):
        values = depth[finite]
        vmin = float(np.percentile(values, 5))
        vmax = float(np.percentile(values, 95))
    else:
        vmin = float(np.nanmin(depth))
        vmax = float(np.nanmax(depth))
    if abs(vmax - vmin) < 1e-6:
        vmax = vmin + 1e-3
    depth_norm = np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = matplotlib.colormaps.get_cmap("magma_r")
    return (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Depth Anything V2 DSFA inference.")
    parser.add_argument("--sample-path", type=Path, default=None, help="JSON or JSONL sample")
    parser.add_argument("--ref-image", type=Path, default=None, help="Reference RGB image")
    parser.add_argument("--stack-images", nargs="+", type=Path, default=None, help="Focus stack images")
    parser.add_argument("--k-values", nargs="+", type=float, default=None, help="K values for stack images")
    parser.add_argument("--stack-indices", nargs="+", type=int, default=None)
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vitl")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--fusion-layers", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--layerscale-init", type=float, default=0.1)
    parser.add_argument("--alibi-scale", type=float, default=1.0)
    parser.add_argument("--use-sdpa", action="store_true")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/depthanythingv2_dsfa"))
    parser.add_argument("--save-numpy", action="store_true")
    parser.add_argument("--pred-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_path is not None:
        ref_path, stack_paths, k_values = read_sample(args.sample_path)
    else:
        if args.ref_image is None or args.stack_images is None or args.k_values is None:
            raise ValueError("--ref-image, --stack-images, and --k-values are required without --sample-path")
        if len(args.stack_images) != len(args.k_values):
            raise ValueError("--stack-images and --k-values must have the same length")
        ref_path = args.ref_image
        stack_paths = list(args.stack_images)
        k_values = [float(item) for item in args.k_values]

    stack_paths, k_values = select_stack(stack_paths, k_values, args.stack_indices)
    ref_image = read_rgb(ref_path)
    stack_images = [read_rgb(path) for path in stack_paths]

    model = DepthAnythingV2DSFA(
        **MODEL_CONFIGS[args.encoder],
        max_depth=args.max_depth,
        fusion_layers=args.fusion_layers,
        num_heads=args.num_heads,
        attn_dropout=args.attn_dropout,
        layerscale_init=args.layerscale_init,
        alibi_scale=args.alibi_scale,
        use_sdpa=args.use_sdpa,
    )
    missing, unexpected = model.load_state_dict(load_state(args.checkpoint), strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    with torch.inference_mode():
        depth = model.infer_image(
            ref_image,
            input_size=args.input_size,
            focus_stack=stack_images,
            k_stack=k_values,
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    color = render_depth(depth)
    color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    color_path = args.outdir / "depth_color.png"
    if args.pred_only:
        cv2.imwrite(str(color_path), color_bgr)
    else:
        spacer = np.full((ref_image.shape[0], 32, 3), 255, dtype=np.uint8)
        color_bgr = cv2.resize(color_bgr, (ref_image.shape[1], ref_image.shape[0]))
        combined = cv2.hconcat([ref_image, spacer, color_bgr])
        cv2.imwrite(str(color_path), combined)

    if args.save_numpy:
        np.save(args.outdir / "depth.npy", depth.astype(np.float32))

    summary = {
        "ref_image": str(ref_path),
        "stack_images": [str(path) for path in stack_paths],
        "k_values": k_values,
        "checkpoint": str(args.checkpoint),
        "depth_color": str(color_path),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote {color_path}")


if __name__ == "__main__":
    main()
