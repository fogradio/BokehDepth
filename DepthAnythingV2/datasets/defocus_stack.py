from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize


IMAGE_KEYS = ("ref", "source_rgb", "image", "image_path", "input_image_path")
DEPTH_KEYS = ("depth", "depth_path", "gt_depth", "target_depth_path")
MASK_KEYS = ("mask", "valid_mask", "mask_path")


def load_manifest(path: str | Path) -> list[dict]:
    manifest_path = Path(path)
    if manifest_path.suffix == ".json":
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return [data]

    samples: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _first_existing(sample: dict, keys: Iterable[str]) -> str | None:
    for key in keys:
        value = sample.get(key)
        if value:
            return str(value)
    return None


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (base_dir / resolved).resolve()


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _read_depth(path: Path, depth_scale: float) -> np.ndarray:
    if path.suffix == ".npy":
        depth = np.load(path).astype(np.float32)
    elif path.suffix == ".npz":
        archive = np.load(path)
        key = "depth" if "depth" in archive else archive.files[0]
        depth = archive[key].astype(np.float32)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Unable to read depth: {path}")
        depth = depth.astype(np.float32)
    return depth * float(depth_scale)


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to read mask: {path}")
    if mask.shape != shape:
        mask = cv2.resize(mask, shape[::-1], interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.float32)


class SharedRandomCrop:
    def __init__(self, size: tuple[int, int]) -> None:
        self.height, self.width = size

    def get_params(self, height: int, width: int) -> tuple[int, int]:
        if height < self.height or width < self.width:
            raise ValueError(
                f"Crop size {(self.height, self.width)} is larger than tensor size {(height, width)}"
            )
        top = random.randint(0, height - self.height)
        left = random.randint(0, width - self.width)
        return top, left

    def apply(self, sample: dict, top: int, left: int) -> dict:
        bottom = top + self.height
        right = left + self.width
        sample["image"] = sample["image"][:, top:bottom, left:right]
        for key in ("depth", "mask"):
            if key in sample:
                sample[key] = sample[key][top:bottom, left:right]
        return sample


class DefocusStackDataset(Dataset):
    """Manifest-backed RGB, depth, and defocus-stack dataset for DSFA training."""

    def __init__(
        self,
        manifest_path: str | Path,
        mode: str = "train",
        size: tuple[int, int] = (518, 518),
        stack_indices: Sequence[int] | None = None,
        depth_scale: float = 1.0,
        min_depth: float = 0.001,
        max_depth: float = 80.0,
    ) -> None:
        if mode not in {"train", "val", "eval"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.manifest_path = Path(manifest_path)
        self.manifest_dir = self.manifest_path.parent
        self.samples = load_manifest(self.manifest_path)
        self.mode = mode
        self.size = size
        self.stack_indices = list(stack_indices) if stack_indices is not None else None
        self.depth_scale = float(depth_scale)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

        width, height = size
        resize_target = mode == "train"
        self.transform = Compose(
            [
                Resize(
                    width=width,
                    height=height,
                    resize_target=resize_target,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                PrepareForNet(),
            ]
        )
        self.crop = SharedRandomCrop((height, width)) if mode == "train" else None

    def __len__(self) -> int:
        return len(self.samples)

    def _stack_from_ids(self, sample: dict) -> tuple[list[Path], list[float]]:
        ids = [str(item) for item in sample["ids"]]
        k_values_raw = sample["k_values"]
        if isinstance(k_values_raw, dict):
            k_values = [float(k_values_raw[item]) for item in ids]
        else:
            k_values = [float(item) for item in k_values_raw]

        stack_dir_value = sample.get("stack_dir") or sample.get("defocus_stack_dir")
        if stack_dir_value is None and "stack_index" in sample:
            stack_dir_value = str(_resolve_path(sample["stack_index"], self.manifest_dir).parent)
        if stack_dir_value is None:
            stack_dir_value = "defocus_stack"

        stack_dir = _resolve_path(stack_dir_value, self.manifest_dir)
        stack_paths = [stack_dir / f"{item}.png" for item in ids]
        return stack_paths, k_values

    def _stack_from_sample(self, sample: dict) -> tuple[list[Path], list[float]]:
        if "stack" in sample and "k" in sample:
            stack_paths = [_resolve_path(item, self.manifest_dir) for item in sample["stack"]]
            k_values = [float(item) for item in sample["k"]]
        elif "ids" in sample and "k_values" in sample:
            stack_paths, k_values = self._stack_from_ids(sample)
        else:
            raise ValueError("Each sample must contain either stack/k or ids/k_values fields")

        if self.stack_indices is None:
            return stack_paths, k_values

        valid_indices = [idx for idx in self.stack_indices if 0 <= idx < len(stack_paths)]
        if not valid_indices:
            raise ValueError("stack_indices did not select any valid stack frames")
        return [stack_paths[idx] for idx in valid_indices], [k_values[idx] for idx in valid_indices]

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image_path_raw = _first_existing(sample, IMAGE_KEYS)
        depth_path_raw = _first_existing(sample, DEPTH_KEYS)
        if image_path_raw is None:
            raise ValueError(f"Sample {index} does not contain an image path")
        if depth_path_raw is None:
            raise ValueError(f"Sample {index} does not contain a depth path")

        image_path = _resolve_path(image_path_raw, self.manifest_dir)
        depth_path = _resolve_path(depth_path_raw, self.manifest_dir)
        image = _read_rgb(image_path)
        depth = _read_depth(depth_path, self.depth_scale)

        mask_path_raw = _first_existing(sample, MASK_KEYS)
        if mask_path_raw:
            mask = _read_mask(_resolve_path(mask_path_raw, self.manifest_dir), depth.shape)
        else:
            mask = np.isfinite(depth).astype(np.float32)

        transformed = self.transform({"image": image, "depth": depth, "mask": mask})
        top = left = 0
        if self.crop is not None:
            height, width = transformed["image"].shape[-2:]
            top, left = self.crop.get_params(height, width)
            transformed = self.crop.apply(transformed, top, left)

        stack_paths, k_values = self._stack_from_sample(sample)
        focus_stack = []
        for stack_path in stack_paths:
            stack_image = _read_rgb(stack_path)
            stack_sample = self.transform({"image": stack_image})
            if self.crop is not None:
                stack_sample = self.crop.apply(stack_sample, top, left)
            focus_stack.append(torch.from_numpy(stack_sample["image"]).contiguous())

        depth_tensor = torch.from_numpy(transformed["depth"]).float().contiguous()
        mask_tensor = torch.from_numpy(transformed["mask"]).float()
        mask_tensor = (
            (mask_tensor > 0.5)
            & torch.isfinite(depth_tensor)
            & (depth_tensor >= self.min_depth)
            & (depth_tensor <= self.max_depth)
        )

        return {
            "image": torch.from_numpy(transformed["image"]).float().contiguous(),
            "depth": depth_tensor,
            "valid_mask": mask_tensor.contiguous(),
            "focus_stack": focus_stack,
            "k_stack": torch.tensor(k_values, dtype=torch.float32),
            "image_path": str(image_path),
        }


def collate_defocus_stack(batch: Sequence[dict]) -> dict:
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    stack_lengths = [len(item["focus_stack"]) for item in batch]
    if len(set(stack_lengths)) != 1:
        raise ValueError(f"All samples in a batch must have the same stack length: {stack_lengths}")

    return {
        "image": torch.stack([item["image"] for item in batch]).contiguous(),
        "depth": torch.stack([item["depth"] for item in batch]).contiguous(),
        "valid_mask": torch.stack([item["valid_mask"] for item in batch]).contiguous(),
        "focus_stack": torch.stack(
            [torch.stack(item["focus_stack"]) for item in batch]
        ).contiguous(),
        "k_stack": torch.stack([item["k_stack"] for item in batch]).unsqueeze(-1).contiguous(),
        "image_path": [item["image_path"] for item in batch],
    }
