"""Utility helpers for the bokeh-generation training datasets.

Two related groups of helpers live here:

1. **Camera-parameter helpers** -- pure math that converts raw EXIF values
   into the canonical conditioning scalar ``dof_cond`` (a.k.a. ``K``) used
   by the bokeh adapter, plus a few related sensor/focal-length utilities.
2. **JSONL dataset helpers** -- dataset-family detection, crop-info
   parsing, target-image loading, and safe foreground-mask / depth-map
   readers. These mirror the JSONL schema documented in ``dataset.py``.
"""

import json
import os
from math import log2

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Camera-parameter helpers
# ---------------------------------------------------------------------------
def extract_N(exif_data):
    """Extract aperture (F-number) from EXIF metadata."""
    try:
        return float(exif_data["FNumber"])
    except ValueError:
        return None


def extract_fmm(exif_data, equiv_35mm=False):
    """Extract focal length (mm) from EXIF; optionally the 35mm-equivalent value."""
    try:
        key = "FocalLengthIn35mmFormat" if equiv_35mm else "FocalLength"
        return float(exif_data[key].replace(" mm", ""))
    except ValueError:
        return None


def extract_iso(exif_data):
    """Extract ISO sensitivity from EXIF metadata."""
    try:
        return float(exif_data["ISOSpeedRatings"])
    except ValueError:
        return None


def calc_crop_factor(fmm, f35mm):
    """Crop factor from real focal length and 35mm equivalent."""
    return f35mm / fmm


def calc_sensor_width(img_width, img_height, fmm, f35mm=50):
    """Estimate physical sensor width (mm) from image aspect ratio and focal length."""
    return ((fmm / f35mm) * 43.27) / np.sqrt(1 + 1 / ((img_width / img_height) ** 2))


def calc_dof_cond(N, fmm, f35mm, s1, img_width, img_height):
    """Compute dof_cond (K value): a scalar conditioning the bokeh strength.

    The result maps the physical depth-of-field quantity into a pixel-scale
    circle-of-confusion radius so it can be fed directly to the bokeh adapter.
    """
    dof_cond = abs((((fmm * 0.001) ** 2) * s1) / (2 * N * (s1 - fmm * 0.001))) * 1000
    sensor_width = calc_sensor_width(img_width, img_height, fmm, f35mm)
    dof_cond = dof_cond * (img_width / sensor_width)
    return dof_cond


def calc_ev_cond(N, iso, shutter_speed):
    """Compute the EV (exposure value) conditioning scalar."""
    ev_cond = log2((100 * (N ** 2)) / (iso * shutter_speed))
    return ev_cond


def fpx_from_f35mm(img_width, img_height, f35mm=50):
    """Rough estimate of focal length in pixels from the 35mm-equivalent value."""
    return f35mm * np.sqrt(img_width ** 2.0 + img_height ** 2.0) / np.sqrt(36 ** 2 + 24 ** 2)


def fpx_from_f35mm_better(img_width, img_height, fmm, f35mm):
    """More accurate focal length in pixels, using the actual sensor width."""
    sensor_width = calc_sensor_width(img_width, img_height, fmm, f35mm)
    return f35mm * (img_width / sensor_width)


# ---------------------------------------------------------------------------
# JSONL dataset helpers
# ---------------------------------------------------------------------------
def detect_dataset_type(jsonl_path: str) -> str:
    """Detect the JSONL dataset family: ITW or BLB.

    Returns one of: ``"blb"``, ``"itw"``.
    """
    with open(jsonl_path, "r") as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError("Empty JSONL file")

        sample = json.loads(first_line)

        # BLB dataset: contains the ``blb_metadata`` field
        if "blb_metadata" in sample:
            return "blb"
        # Everything else is treated as the in-the-wild (ITW) collection
        return "itw"


def get_crop_info(sample: dict) -> dict:
    """Return the FLUX crop-info dict from a sample."""
    crop_info = sample["crop_info"]
    if "flux" not in crop_info:
        raise ValueError(
            f"crop_info missing 'flux' key. Available keys: {list(crop_info.keys())}"
        )
    return crop_info["flux"]


def load_target_image_for_blb(batch_or_sample):
    """Load the pre-rendered target bokeh image for BLB-style samples.

    ``batch_or_sample`` may be either a single sample dict or a collated batch.
    All ``target_image_path`` entries are expected to be absolute paths.
    """
    # Batch form
    if (
        isinstance(batch_or_sample, dict)
        and "target_image_path" in batch_or_sample
        and isinstance(batch_or_sample["target_image_path"], list)
    ):
        batch = batch_or_sample
        target_images = []
        for target_path in batch["target_image_path"]:
            if not os.path.exists(target_path):
                raise FileNotFoundError(f"Target image not found: {target_path}")
            target_images.append(Image.open(target_path).convert("RGB"))
        return target_images

    # Single-sample form
    sample = batch_or_sample
    if "target_image_path" not in sample:
        raise ValueError("BLB sample missing target_image_path")

    target_path = sample["target_image_path"]
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target image not found: {target_path}")
    return Image.open(target_path).convert("RGB")


def load_foreground_mask_safe(sample: dict):
    """Load the foreground mask; return ``None`` if the field is missing/empty."""
    fg_mask_path = sample.get("fg_mask_path", "")
    if fg_mask_path and os.path.exists(fg_mask_path):
        return np.array(Image.open(fg_mask_path).convert("L"))
    return None


def get_target_size_from_sample(sample: dict) -> int:
    """Return the target side length (assumed square) recorded in ``crop_info``."""
    crop_info = get_crop_info(sample)
    return crop_info["crop_size"][0]


def should_use_bokehme(sample: dict) -> bool:
    """Whether the sample needs on-the-fly BokehMe rendering.

    BLB is already pre-rendered, so it skips BokehMe.
    ITW samples require on-line rendering.
    """
    if "blb_metadata" in sample:
        return False
    return True


def load_depth_map_safe(sample: dict) -> np.ndarray:
    """Safely load a depth map, auto-detecting ``.npz`` / ``.jpg`` / ``.png``.

    For disparity-encoded depth (BLB uses ``disparity.jpg``), the function does
    a simple ``1 / (disp + eps)`` inverse mapping and rescales the result to
    the [0.1, 100] range so it is roughly comparable with NPZ depth maps.
    """
    depth_path = sample["depth_map_path"]
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth map not found: {depth_path}")

    if depth_path.endswith(".npz"):
        return np.load(depth_path)["depth"]
    if depth_path.endswith(".jpg") or depth_path.endswith(".png"):
        disparity_img = Image.open(depth_path).convert("L")
        disparity_array = np.array(disparity_img, dtype=np.float32)

        disparity_normalized = disparity_array / 255.0
        epsilon = 0.01
        depth = 1.0 / (disparity_normalized + epsilon)

        depth_min, depth_max = 0.1, 100.0
        depth_scaled = (
            depth_min
            + (depth - depth.min()) / (depth.max() - depth.min()) * (depth_max - depth_min)
        )
        return depth_scaled
    raise ValueError(f"Unsupported depth map format: {depth_path}")


def print_dataset_info(jsonl_path: str):
    """Print dataset statistics so users can sanity-check a new JSONL."""
    dataset_type = detect_dataset_type(jsonl_path)
    task_counts = {}
    total_samples = 0

    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line.strip())
                task_type = sample.get("task_type", "unknown")
                task_counts[task_type] = task_counts.get(task_type, 0) + 1
                total_samples += 1

    print("\n=== Dataset info ===")
    print(f"Dataset type : {dataset_type.upper()}")
    print(f"Sample count : {total_samples}")
    print("Task distribution:")
    for task_type, count in task_counts.items():
        print(f"  {task_type}: {count} ({count / total_samples * 100:.1f}%)")
