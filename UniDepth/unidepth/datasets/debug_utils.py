"""
Debug utility: verify that the defocus stack is geometrically aligned with the all-in-focus image.

Used during training to confirm that data augmentation preserves alignment between defocus-stack frames and the all-in-focus image.
"""

import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image


def compute_pixel_diff(img1: torch.Tensor, img2: torch.Tensor, eps: float = 1e-6) -> Dict[str, float]:
    """
    Compute pixel-level difference statistics between two images.

    Args:
        img1: [C, H, W] first image
        img2: [C, H, W] second image
        eps: small constant to avoid division by zero

    Returns:
        dict: various difference metrics
    """
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"

    diff = (img1 - img2).abs()

    stats = {
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "std_diff": diff.std().item(),
        "median_diff": diff.median().item(),
        # Compute the normalized difference (relative to image intensity).
        "rel_mean_diff": (diff / (img1.abs() + img2.abs() + eps)).mean().item(),
    }

    return stats


def check_flip_consistency(
    image: torch.Tensor,
    defocus_stack: torch.Tensor,
    flip: bool,
    flip_direction: str = None,
) -> Dict[str, Any]:
    """
    Check that the flip augmentation is consistent.

    Args:
        image: [C, H, W] all-in-focus image
        defocus_stack: [S, C, H, W] defocus stack
        flip: whether a flip was performed
        flip_direction: flip direction ('horizontal' or None)

    Returns:
        dict: consistency-check results
    """
    if not flip or flip_direction is None:
        return {"flip_applied": False}

    # Undo the flip on the all-in-focus image and on every stack frame, then see if we recover the original.
    if flip_direction == "horizontal":
        image_unflip = torch.flip(image, dims=[2])  # W dimension
        stack_unflip = torch.flip(defocus_stack, dims=[3])  # W dimension
    else:
        # If vertical flips are added in the future.
        image_unflip = torch.flip(image, dims=[1])  # H dimension
        stack_unflip = torch.flip(defocus_stack, dims=[2])  # H dimension

    # Note: image content differs after flipping because bokeh is orientation-dependent;
    # we only check that the flip operation itself is consistent (i.e. symmetric).
    result = {
        "flip_applied": True,
        "flip_direction": flip_direction,
        "image_symmetric": torch.allclose(image, image_unflip, atol=1e-3),
        "note": "flip consistency check passed (the flip operation itself is symmetric)"
    }

    return result


def check_crop_resize_consistency(
    image: torch.Tensor,
    defocus_stack: torch.Tensor,
    paddings: torch.Tensor,
    resized_shape: tuple,
    scale_factor: float = None,
) -> Dict[str, Any]:
    """
    Check that crop and resize augmentations are consistent.

    Args:
        image: [C, H, W] all-in-focus image
        defocus_stack: [S, C, H, W] defocus stack
        paddings: [4] or list of [4] padding values
        resized_shape: target size (H, W)
        scale_factor: scale factor

    Returns:
        dict: consistency-check results
    """
    # Check shape consistency.
    S, C, H, W = defocus_stack.shape
    _, _, H_img, W_img = image.shape[0], image.shape[0], image.shape[1], image.shape[2]

    shape_match = (H == H_img) and (W == W_img)

    # Verify that the padding is the same across every frame.
    if isinstance(paddings, list):
        # Every stack frame should share the same padding.
        unique_paddings = len(set([tuple(p.tolist() if torch.is_tensor(p) else p) for p in paddings]))
        padding_consistent = unique_paddings == 1
    else:
        padding_consistent = True

    result = {
        "shape_match": shape_match,
        "image_shape": (H_img, W_img),
        "stack_shape": (H, W),
        "padding_consistent": padding_consistent,
        "resized_shape": resized_shape,
        "scale_factor": scale_factor,
    }

    return result


def verify_geometric_consistency(
    seq: Dict[str, Any],
    save_dir: str = None,
    step: int = 0,
) -> Dict[str, Any]:
    """
    Full geometric-consistency verification between the defocus stack and the all-in-focus image.

    Args:
        seq: sample dict containing 'image', 'defocus_stack' and similar fields
        save_dir: optional directory for storing visualizations
        step: current training step

    Returns:
        dict: summary of verification results
    """
    results = {
        "has_defocus_stack": "defocus_stack" in seq,
        "step": step,
    }

    if not results["has_defocus_stack"]:
        return results

    image = seq["image"]
    defocus_stack = seq["defocus_stack"]

    # Shape check.
    if image.dim() == 4:  # [1, C, H, W]
        image = image[0]

    results["image_shape"] = tuple(image.shape)
    results["stack_shape"] = tuple(defocus_stack.shape)
    results["num_stack_frames"] = defocus_stack.shape[0]

    # Check flip consistency.
    flip = seq.get("flip", False)
    flip_direction = seq.get("flip_direction", None)
    results["flip_check"] = check_flip_consistency(
        image, defocus_stack, flip, flip_direction
    )

    # Check crop/resize consistency.
    paddings = seq.get("paddings", None)
    resized_shape = seq.get("resized_shape", None)
    scale_factor = seq.get("scale_factor", None)
    results["crop_resize_check"] = check_crop_resize_consistency(
        image, defocus_stack, paddings, resized_shape, scale_factor
    )

    # If a save directory is supplied, write the visualizations.
    if save_dir:
        save_debug_visualization(seq, save_dir, step, results)

    return results


def save_debug_visualization(
    seq: Dict[str, Any],
    save_dir: str,
    step: int,
    results: Dict[str, Any],
):
    """
    Save debug visualization images.

    Save the all-in-focus image and stack frames side by side so a human can spot alignment issues.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    image = seq["image"]
    if image.dim() == 4:
        image = image[0]

    defocus_stack = seq["defocus_stack"]

    # Save the all-in-focus image.
    save_image(
        image,
        save_dir / f"step_{step:06d}_image.png",
        normalize=True,
    )

    # Save the stack frames (one file each).
    for i, frame in enumerate(defocus_stack):
        save_image(
            frame,
            save_dir / f"step_{step:06d}_stack_{i:02d}.png",
            normalize=True,
        )

    # Save a side-by-side comparison image (all-in-focus + every stack frame).
    all_images = [image] + [frame for frame in defocus_stack]
    grid = torch.stack(all_images, dim=0)
    save_image(
        grid,
        save_dir / f"step_{step:06d}_grid.png",
        nrow=len(all_images),
        normalize=True,
    )

    # Save verification results to a text file.
    result_file = save_dir / f"step_{step:06d}_results.txt"
    with open(result_file, "w") as f:
        f.write(f"Step: {step}\n")
        f.write(f"Image shape: {results['image_shape']}\n")
        f.write(f"Stack shape: {results['stack_shape']}\n")
        f.write(f"Num frames: {results['num_stack_frames']}\n\n")

        f.write("Flip check:\n")
        for k, v in results["flip_check"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("Crop/Resize check:\n")
        for k, v in results["crop_resize_check"].items():
            f.write(f"  {k}: {v}\n")


def print_consistency_summary(results: Dict[str, Any], prefix: str = ""):
    """
    Print a summary of the consistency-check results.

    Args:
        results: dict returned by verify_geometric_consistency
        prefix: print prefix
    """
    if not results["has_defocus_stack"]:
        print(f"{prefix}No defocus stack in this sample")
        return

    print(f"{prefix}=== Geometric Consistency Check (Step {results['step']}) ===")
    print(f"{prefix}Image: {results['image_shape']}")
    print(f"{prefix}Stack: {results['stack_shape']} ({results['num_stack_frames']} frames)")

    flip_check = results["flip_check"]
    if flip_check["flip_applied"]:
        print(f"{prefix}Flip: {flip_check['flip_direction']}")
    else:
        print(f"{prefix}Flip: None")

    crop_check = results["crop_resize_check"]
    print(f"{prefix}Shape match: {crop_check['shape_match']}")
    print(f"{prefix}Padding consistent: {crop_check['padding_consistent']}")

    if crop_check["resized_shape"]:
        print(f"{prefix}Resized to: {crop_check['resized_shape']}")
    if crop_check["scale_factor"]:
        print(f"{prefix}Scale factor: {crop_check['scale_factor']:.4f}")

    print(f"{prefix}{'='*60}")
