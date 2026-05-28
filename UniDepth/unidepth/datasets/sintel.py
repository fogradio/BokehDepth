"""Sintel manifest-based dataset loader with defocus stack support."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from unidepth.utils import eval_depth, sync_tensor_across_gpus
from unidepth.utils.camera import BatchCamera, Pinhole
from unidepth.utils.constants import IMAGENET_DATASET_MEAN, IMAGENET_DATASET_STD

LOGGER = logging.getLogger("unidepth.datasets.sintel")

SINTEL_TAG_FLOAT = 202021.25
SINTEL_MIN_DEPTH = 0.0
SINTEL_MAX_DEPTH = 80.0


@dataclass
class SintelSample:
    image_path: Path
    depth_path: Path
    intrinsics_path: Optional[Path]
    focus_stack: List[Path]
    focus_values: List[float]
    scene: str
    frame: str
    source_rgb_clean: Optional[Path]
    source_rgb_final: Optional[Path]


def _read_sintel_depth(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Sintel depth file not found: {path}")
    with path.open("rb") as handle:
        tag = np.fromfile(handle, dtype=np.float32, count=1)
        if tag.size == 0 or float(tag[0]) != SINTEL_TAG_FLOAT:
            raise ValueError(f"Invalid Sintel depth tag in {path}")
        width = int(np.fromfile(handle, dtype=np.int32, count=1)[0])
        height = int(np.fromfile(handle, dtype=np.int32, count=1)[0])
        size = width * height
        depth = np.fromfile(handle, dtype=np.float32, count=size)
    if depth.size != size:
        raise ValueError(f"Unexpected depth payload in {path}: expected {size}, got {depth.size}")
    depth = depth.reshape((height, width)).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def _read_sintel_intrinsics(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Sintel camera file not found: {path}")
    with path.open("rb") as handle:
        tag = np.fromfile(handle, dtype=np.float32, count=1)
        if tag.size == 0 or float(tag[0]) != SINTEL_TAG_FLOAT:
            raise ValueError(f"Invalid Sintel camera tag in {path}")
        intrinsics = np.fromfile(handle, dtype=np.float64, count=9)
        if intrinsics.size != 9:
            raise ValueError(f"Malformed Sintel intrinsics in {path}")
        # Remaining 3x4 extrinsics entries are ignored for now
    return intrinsics.reshape(3, 3).astype(np.float32)


def _scale_intrinsics(intrinsics: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    scaled = intrinsics.copy()
    scaled[0, 0] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[0, 2] *= scale_x
    scaled[1, 2] *= scale_y
    return scaled


def _load_rgb(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Sintel RGB file not found: {path}")
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read Sintel RGB image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _load_focus_stack(paths: Iterable[Path]) -> List[np.ndarray]:
    stack: List[np.ndarray] = []
    for item in paths:
        if not item:
            continue
        image_bgr = cv2.imread(str(item), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Failed to read Sintel focus stack frame: {item}")
        stack.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    return stack


class SintelBokehDataset(Dataset):
    """Dataset that reads Sintel manifest entries including defocus stacks."""

    min_depth = SINTEL_MIN_DEPTH
    max_depth = SINTEL_MAX_DEPTH

    def __init__(
        self,
        manifest_path: str | Path,
        mode: str = "final",
        img_size: Optional[int] = None,
        stack_indices: Optional[Sequence[int]] = None,
        shape_mult: int = 14,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if mode.lower() not in {"clean", "final"}:
            raise ValueError("Sintel mode must be either 'clean' or 'final'")
        self.mode = mode.lower()
        self.img_size = img_size
        self.stack_indices = list(stack_indices) if stack_indices is not None else None
        if shape_mult <= 0:
            raise ValueError("shape_mult must be positive")
        self.shape_mult = int(shape_mult)
        self.samples = self._load_manifest()

        # Initialize attributes needed for evaluation.
        self.metrics_store = {}
        self.metrics_count = {}
        self.image_mean = torch.tensor(
            IMAGENET_DATASET_MEAN, dtype=torch.float32
        ).view(3, 1, 1)
        self.image_std = torch.tensor(
            IMAGENET_DATASET_STD, dtype=torch.float32
        ).view(3, 1, 1)
        self.normalization_stats = {
            "mean": self.image_mean.view(3).clone(),
            "std": self.image_std.view(3).clone(),
        }

    def _load_manifest(self) -> List[SintelSample]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest does not exist: {self.manifest_path}")
        samples: List[SintelSample] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                record = json.loads(payload)
                ref_path = Path(record.get("ref", ""))
                depth_path = Path(record.get("depth", ""))
                intrinsics_path = Path(record["source_intrinsics"]) if record.get("source_intrinsics") else None
                stack_paths = [Path(p) for p in record.get("stack", []) or []]
                scene = record.get("scene", "")
                frame = str(record.get("frame", ""))
                sample = SintelSample(
                    image_path=ref_path,
                    depth_path=depth_path,
                    intrinsics_path=intrinsics_path,
                    focus_stack=stack_paths,
                    focus_values=[float(v) for v in record.get("k", []) or []],
                    scene=scene,
                    frame=frame,
                    source_rgb_clean=Path(record["source_rgb_clean"]) if record.get("source_rgb_clean") else None,
                    source_rgb_final=Path(record["source_rgb_final"]) if record.get("source_rgb_final") else None,
                )
                samples.append(sample)
        if not samples:
            raise RuntimeError(f"No entries found in manifest {self.manifest_path}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _resize_if_needed(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        focus_stack: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], float, float]:
        if self.img_size is None:
            return image, depth, mask, focus_stack, 1.0, 1.0

        target = int(self.img_size)
        h, w = image.shape[:2]
        scale = min(target / h, target / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        resized_depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        resized_mask = cv2.resize(mask.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0.5
        resized_stack = [
            cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            for frame in focus_stack
        ]
        scale_x = new_w / w
        scale_y = new_h / h
        return resized_image, resized_depth, resized_mask, resized_stack, scale_x, scale_y

    def _pad_to_shape_mult(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        focus_stack: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], int, int]:
        h, w = image.shape[:2]
        target_h = int(math.ceil(h / self.shape_mult) * self.shape_mult)
        target_w = int(math.ceil(w / self.shape_mult) * self.shape_mult)
        pad_bottom = target_h - h
        pad_right = target_w - w
        if pad_bottom == 0 and pad_right == 0:
            return image, depth, mask, focus_stack, 0, 0

        image_padded = np.pad(
            image,
            ((0, pad_bottom), (0, pad_right), (0, 0)),
            mode="edge",
        )
        depth_padded = np.pad(
            depth,
            ((0, pad_bottom), (0, pad_right)),
            mode="constant",
            constant_values=0.0,
        )
        mask_padded = np.pad(
            mask.astype(np.bool_),
            ((0, pad_bottom), (0, pad_right)),
            mode="constant",
            constant_values=False,
        )
        if focus_stack:
            focus_padded = [
                np.pad(
                    frame,
                    ((0, pad_bottom), (0, pad_right), (0, 0)),
                    mode="edge",
                )
                for frame in focus_stack
            ]
        else:
            focus_padded = focus_stack

        return (
            image_padded,
            depth_padded,
            mask_padded.astype(np.bool_),
            focus_padded,
            pad_right,
            pad_bottom,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | List[str] | str | List[Path]]:
        sample = self.samples[idx]
        if self.mode == "final" and sample.source_rgb_final:
            rgb_path = sample.source_rgb_final
        elif self.mode == "clean" and sample.source_rgb_clean:
            rgb_path = sample.source_rgb_clean
        else:
            rgb_path = sample.image_path

        # Load the reference image.
        image_rgb = _load_rgb(rgb_path).astype(np.float32)
        ref_height, ref_width = image_rgb.shape[:2]  # record reference-image size

        depth_original = _read_sintel_depth(sample.depth_path)
        # Clip depth to valid range (consistent with Depth-Anything-V2)
        depth_original = np.clip(depth_original, self.min_depth, self.max_depth).astype(np.float32)
        mask_original = (depth_original > self.min_depth) & (depth_original < self.max_depth)

        # Load defocus-stack paths.
        focus_paths = sample.focus_stack
        if self.stack_indices is not None:
            focus_paths = [focus_paths[i] for i in self.stack_indices if 0 <= i < len(focus_paths)]

        # Load defocus-stack images and make sure they match the reference size (see eth3d.py's implementation).
        focus_images: List[np.ndarray] = []
        for path in focus_paths:
            if not path:
                continue
            focus_img = _load_rgb(path)  # same RGB loader

            # Resize if the stack image differs from the reference size.
            if focus_img.shape[:2] != (ref_height, ref_width):
                focus_img = cv2.resize(
                    focus_img,
                    (ref_width, ref_height),  # cv2.resize takes (width, height)
                    interpolation=cv2.INTER_CUBIC
                )
            focus_images.append(focus_img.astype(np.float32))

        if not focus_images:
            LOGGER.warning("Sintel sample missing focus stack, skipping sample %s/%s", sample.scene, sample.frame)
            return None

        # 1. Uniformly resize every piece of data.
        image_resized, depth_resized, mask_resized, focus_resized, scale_x, scale_y = self._resize_if_needed(
            image_rgb,
            depth_original,
            mask_original,
            focus_images,
        )

        if not focus_resized and focus_images:
            focus_resized = focus_images

        # 2. Pad once so every tensor is aligned.
        (
            image_padded,
            depth_padded,
            mask_padded,
            focus_padded,
            pad_right,
            pad_bottom,
        ) = self._pad_to_shape_mult(
            image_resized,
            depth_resized,
            mask_resized,
            focus_resized,
        )

        # 3. Handle intrinsics.
        intrinsics = _read_sintel_intrinsics(sample.intrinsics_path)
        if intrinsics is None:
            intrinsics = np.eye(3, dtype=np.float32)
        else:
            intrinsics = intrinsics.astype(np.float32)
        intrinsics_scaled = _scale_intrinsics(intrinsics, scale_x, scale_y)

        # 4. Convert to tensors with the same normalization.
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(image_padded))
            .permute(2, 0, 1)
            .contiguous()
            .to(torch.float32)
            / 255.0
        )
        image_tensor = (image_tensor - self.image_mean) / self.image_std

        depth_tensor = torch.from_numpy(depth_padded.astype(np.float32))
        mask_tensor = torch.from_numpy(mask_padded.astype(np.bool_))

        # 5. Process the focus stack with the exact same preprocessing as the image.
        focus_tensor = None
        if focus_padded:
            focus_frames: List[torch.Tensor] = []
            for frame in focus_padded:
                frame_tensor = (
                    torch.from_numpy(np.ascontiguousarray(frame))
                    .permute(2, 0, 1)
                    .contiguous()
                    .to(torch.float32)
                    / 255.0
                )
                frame_tensor = (frame_tensor - self.image_mean) / self.image_std
                focus_frames.append(frame_tensor)
            focus_tensor = torch.stack(focus_frames, dim=0)  # [K, C, H, W]

            # Verify size consistency.
            assert focus_tensor.shape[-2:] == image_tensor.shape[-2:], \
                f"Shape mismatch after padding: focus {focus_tensor.shape} vs image {image_tensor.shape}"

        # 6. Handle k_values.
        k_values = sample.focus_values
        if self.stack_indices is not None and k_values:
            k_values = [k_values[i] for i in self.stack_indices if 0 <= i < len(k_values)]

        if focus_tensor is not None:
            num_focus_frames = focus_tensor.shape[0]
            if not k_values:
                k_tensor = torch.zeros(num_focus_frames, dtype=torch.float32)
            else:
                k_tensor = torch.tensor(k_values, dtype=torch.float32)
                if k_tensor.shape[0] != num_focus_frames:
                    # If k_values has the wrong length, fall back to defaults.
                    LOGGER.warning(
                        f"Focus stack length mismatch for {sample.scene}/{sample.frame}: "
                        f"k={k_tensor.shape[0]} vs stack={num_focus_frames}, using zeros"
                    )
                    k_tensor = torch.zeros(num_focus_frames, dtype=torch.float32)
        else:
            k_tensor = torch.zeros(0, dtype=torch.float32)

        # 7. Build the output, keeping the same shape as the standard mode.
        resized_shape = [list(image_tensor.shape[-2:])]
        paddings = [[0, pad_right, 0, pad_bottom]]
        depth_paddings = [[0, pad_right, 0, pad_bottom]]
        image_ori_shape = [list(image_rgb.shape[:2])]

        output: Dict[str, torch.Tensor | str | List[str] | List[int]] = {
            "image": image_tensor,
            "depth": depth_tensor,
            "valid_mask": mask_tensor,
            "intrinsics": torch.from_numpy(intrinsics_scaled),
            "original_depth": torch.from_numpy(depth_original.astype(np.float32)),
            "original_valid_mask": torch.from_numpy(mask_original.astype(np.bool_)),
            "original_intrinsics": torch.from_numpy(intrinsics),
            "scene": sample.scene,
            "frame": sample.frame,
            "image_path": str(rgb_path),
            "depth_path": str(sample.depth_path),
            "paddings": paddings,
            "depth_paddings": depth_paddings,
            "resized_shape": resized_shape,
            "image_ori_shape": image_ori_shape,
        }

        if focus_tensor is not None:
            output["focus_stack"] = focus_tensor
        output["k_stack"] = k_tensor

        return output

    def prepare_depth_eval(self, inputs, preds):
        """Prepare data for depth evaluation."""
        depth_gt = inputs["depth"]
        depth_pred = preds["depth"]
        depth_masks = inputs.get(
            "depth_mask", torch.ones_like(depth_gt, dtype=torch.bool)
        )

        if depth_gt.dim() == 5:
            depth_gt = depth_gt[:, 0]
        if depth_pred.dim() == 5:
            depth_pred = depth_pred[:, 0]
        if depth_masks.dim() == 5:
            depth_masks = depth_masks[:, 0]

        if depth_gt.dim() == 3:
            depth_gt = depth_gt.unsqueeze(1)
        if depth_pred.dim() == 3:
            depth_pred = depth_pred.unsqueeze(1)
        if depth_masks.dim() == 3:
            depth_masks = depth_masks.unsqueeze(1)

        depth_masks = depth_masks.bool()

        return depth_gt, {"depth": depth_pred}, depth_masks

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def accumulate_metrics_depth(self, gts, preds, masks):
        """Accumulate depth evaluation metrics."""
        for eval_type, pred in preds.items():
            log_name = eval_type.replace("depth", "").strip("-").strip("_")
            if log_name not in self.metrics_store:
                self.metrics_store[log_name] = {}
            current_count = self.metrics_count.get(
                log_name, torch.tensor([], device=gts.device)
            )
            new_count = masks.view(gts.shape[0], -1).sum(dim=-1)
            self.metrics_count[log_name] = torch.cat([current_count, new_count])
            for k, v in eval_depth(gts, pred, masks, max_depth=self.max_depth).items():
                current_metric = self.metrics_store[log_name].get(
                    k, torch.tensor([], device=gts.device)
                )
                self.metrics_store[log_name][k] = torch.cat(
                    [current_metric, v.reshape(-1)]
                )

    def accumulate_metrics(self, inputs, preds, keyframe_idx=None, metrics=["depth"]):
        """Accumulate eval metrics - BaseDataset-compatible interface."""
        if "depth" in metrics:
            depth_gt, depth_pred, depth_masks = self.prepare_depth_eval(inputs, preds)
            self.accumulate_metrics_depth(depth_gt, depth_pred, depth_masks)

    def get_evaluation(self, metrics=None):
        """Return eval results - BaseDataset-compatible interface."""
        metric_vals = {}
        for eval_type in metrics if metrics is not None else self.metrics_store.keys():
            if not self.metrics_store.get(eval_type):
                continue
            cnts = sync_tensor_across_gpus(self.metrics_count[eval_type])
            for name, val in self.metrics_store[eval_type].items():
                vals_r = sync_tensor_across_gpus(val).mean()
                metric_vals[f"{eval_type}_{name}".strip("_")] = np.round(
                    vals_r.cpu().item(), 5
                )
            self.metrics_store[eval_type] = {}
        self.metrics_count = {}
        return metric_vals


def collate_fn_sintel_bokeh(batch: Sequence[Dict[str, torch.Tensor | str]]) -> Dict[str, Any]:
    """Collate function compatible with UniDepth validation pipeline.

    Returns a dictionary with 'data' and 'img_metas' keys to match the expected format.
    """
    clean_batch = [item for item in batch if item is not None]
    if not clean_batch:
        raise ValueError("Sintel batch is empty or all samples invalid")

    images = torch.stack([item["image"] for item in clean_batch])  # [B, C, H, W]
    depths = torch.stack([item["depth"] for item in clean_batch])  # [B, H, W]
    masks = torch.stack([item["valid_mask"] for item in clean_batch])  # [B, H, W]
    intrinsics = torch.stack([item["intrinsics"] for item in clean_batch])  # [B, 3, 3]
    k_stack = torch.stack([item["k_stack"] for item in clean_batch])  # [B, K]

    # Add a time dimension (T=1) to match the training pipeline format.
    images = images.unsqueeze(1)  # [B, 1, C, H, W]
    depths = depths.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, H, W]
    masks = masks.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, H, W]

    # Create BatchCamera object from intrinsics matrices
    # intrinsics shape: [B, 3, 3]
    # Note: validation.py will call squeeze(1) on all fields, but camera
    # should remain as BatchCamera object. We don't add temporal dimension
    # to camera since squeeze(1) won't affect [B, 3, 3] shape.
    batch_camera = BatchCamera.from_camera(Pinhole(K=intrinsics))

    # Build the data dict
    data_dict = {
        "image": images,
        "depth": depths,
        "depth_mask": masks,
        "camera": batch_camera,  # Now it's a BatchCamera object
    }

    # Attach the focus stack if available.
    if all("focus_stack" in item for item in clean_batch):
        stack_lengths = [item["focus_stack"].shape[0] for item in clean_batch]
        if not all(length == stack_lengths[0] for length in stack_lengths):
            raise ValueError(f"Sintel focus stack length mismatch: {stack_lengths}")
        focus_stack = torch.stack([item["focus_stack"] for item in clean_batch])  # [B, K, C, H, W]
        data_dict["defocus_stack"] = focus_stack.unsqueeze(1)  # [B, 1, K, C, H, W]
        data_dict["k_values"] = k_stack.unsqueeze(1)  # [B, 1, K]

    # Build img_metas with metadata
    img_metas = []
    for item in clean_batch:
        meta = {
            "scene": item.get("scene", ""),
            "frame": item.get("frame", ""),
            "image_path": item.get("image_path", ""),
            "depth_path": item.get("depth_path", ""),
            "original_depth": item["original_depth"],
            "original_valid_mask": item["original_valid_mask"],
            "original_intrinsics": item["original_intrinsics"],
        }
        if "paddings" in item:
            meta["paddings"] = item["paddings"]
        if "depth_paddings" in item:
            meta["depth_paddings"] = item["depth_paddings"]
        if "resized_shape" in item:
            meta["resized_shape"] = item["resized_shape"]
        if "image_ori_shape" in item:
            meta["image_ori_shape"] = item["image_ori_shape"]
        img_metas.append(meta)

    return {
        "data": data_dict,
        "img_metas": img_metas,
    }


# Backwards compatibility alias used by previous code paths.
Sintel = SintelBokehDataset

__all__ = ["SintelBokehDataset", "collate_fn_sintel_bokeh", "Sintel"]
