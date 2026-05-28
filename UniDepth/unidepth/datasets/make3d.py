"""Make3D dataset loader with manifest + defocus stack support."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.io import read_image

from unidepth.datasets.image_dataset import ImageDataset
from unidepth.datasets.utils import DatasetFromList
from unidepth.utils.camera import BatchCamera, Pinhole

try:  # scipy is required only when reading depth .mat files
    from scipy.io import loadmat
except ImportError:  # pragma: no cover - defensive fallback
    loadmat = None  # type: ignore[misc]


class Make3D(ImageDataset):
    """Manifest-driven Make3D dataset supporting defocus stacks."""

    min_depth = 0.1
    max_depth = 70.0
    c1_max_depth = 70.0
    c2_max_depth = 80.0
    depth_scale = 1.0
    train_split = "test.txt"
    test_split = "test.txt"

    def __init__(
        self,
        image_shape,
        split_file,
        test_mode,
        benchmark: bool = False,
        augmentations_db: Optional[Dict[str, Any]] = None,
        normalize: bool = True,
        resize_method: str = "hard",
        mini: float = 1.0,
        **kwargs,
    ) -> None:
        augmentations_db = augmentations_db or {}

        manifest_path = kwargs.pop("manifest_path", None) or os.environ.get("MAKE3D_MANIFEST_PATH")
        manifest_split_path = kwargs.pop("manifest_split_path", None)
        defocus_stack_indices = kwargs.pop("defocus_stack_indices", None)
        debug_augmentation = kwargs.pop("debug_augmentation", False)
        evaluation_mode = kwargs.pop("evaluation_mode", "c1")
        depth_channel = int(kwargs.pop("depth_channel", 3))
        max_depth_clip = float(kwargs.pop("max_depth_clip", self.c2_max_depth))
        data_root = kwargs.pop("data_root", None)

        if defocus_stack_indices is not None:
            if isinstance(defocus_stack_indices, (str, bytes)):
                indices = str(defocus_stack_indices).split(",")
                defocus_stack_indices = [int(idx) for idx in indices if idx]
            self.defocus_stack_indices = [int(i) for i in defocus_stack_indices]
        else:
            self.defocus_stack_indices = None

        self.manifest_path = manifest_path
        self.manifest_split_path = manifest_split_path
        self.debug_augmentation = bool(debug_augmentation)
        self.depth_channel = int(depth_channel)
        self.max_depth_clip = float(max_depth_clip)
        self.eval_protocol = str(evaluation_mode).lower()
        if self.eval_protocol not in {"c1", "c2"}:
            raise ValueError(f"Unsupported Make3D evaluation_mode: {evaluation_mode}")

        # Cap evaluation depth according to protocol before base initialisation
        if self.eval_protocol == "c1":
            self.max_depth = min(self.c1_max_depth, self.max_depth_clip)
        else:
            self.max_depth = min(self.c2_max_depth, self.max_depth_clip)

        self._use_manifest = False
        self._manifest_split_whitelist: Optional[set[str]] = None

        super().__init__(
            image_shape=image_shape,
            split_file=split_file,
            test_mode=test_mode,
            benchmark=benchmark,
            normalize=normalize,
            augmentations_db=augmentations_db,
            resize_method=resize_method,
            mini=mini,
            data_root=data_root,
            **kwargs,
        )
        self.test_mode = test_mode
        self.load_dataset()

    def load_dataset(self) -> None:
        manifest_path = self.manifest_path
        if not manifest_path:
            raise FileNotFoundError(
                "Make3D manifest_path was not provided. Set --manifest-path or MAKE3D_MANIFEST_PATH."
            )
        manifest_path = self._resolve_path(manifest_path)
        if not manifest_path or not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        self._load_manifest_dataset(manifest_path)

    def _load_manifest_dataset(self, manifest_path: str) -> None:
        records: List[Dict[str, Any]] = []
        split_whitelist = self._read_manifest_split_whitelist()

        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if split_whitelist and not self._manifest_entry_in_split(entry, split_whitelist):
                    continue

                record = self._build_manifest_sample(entry)
                if record:
                    records.append(record)

        if not records:
            raise FileNotFoundError(
                f"Parsed manifest {manifest_path} but no valid Make3D samples were found."
            )

        if not self.test_mode:
            records = self.chunk(records, chunk_dim=1, pct=self.mini)

        self.dataset = DatasetFromList(records, serialize=False)
        self._use_manifest = True
        self.manifest_path = manifest_path
        self.depth_scale = 1.0
        self.log_load_dataset()

    def _build_manifest_sample(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        image_path = entry.get("ref") or entry.get("image")
        depth_path = entry.get("depth")
        if not image_path or not depth_path:
            return None

        image_path = self._resolve_path(image_path)
        depth_path = self._resolve_path(depth_path)
        if not image_path or not depth_path:
            return None
        if not os.path.isfile(image_path) or not os.path.isfile(depth_path):
            return None

        stack_paths: List[str] = []
        for stack_item in entry.get("stack", []):
            resolved = self._resolve_path(stack_item)
            if resolved and os.path.isfile(resolved):
                stack_paths.append(resolved)

        k_values = entry.get("k", [])
        if self.defocus_stack_indices and stack_paths:
            filtered_paths: List[str] = []
            filtered_k: List[float] = []
            for idx in self.defocus_stack_indices:
                if 0 <= idx < len(stack_paths):
                    filtered_paths.append(stack_paths[idx])
                    if k_values and idx < len(k_values):
                        filtered_k.append(k_values[idx])
            if filtered_paths:
                stack_paths = filtered_paths
                if k_values:
                    k_values = filtered_k if filtered_k else []

        stack_index = entry.get("stack_index")
        if stack_index:
            stack_index = self._resolve_path(stack_index)

        source_rgb = entry.get("source_rgb")
        if source_rgb:
            source_rgb = self._resolve_path(source_rgb)
        source_depth = entry.get("source_depth")
        if source_depth:
            source_depth = self._resolve_path(source_depth)

        return {
            "image_path": image_path,
            "depth_path": depth_path,
            "stack_paths": stack_paths,
            "k_values": k_values,
            "stack_index": stack_index,
            "source_rgb": source_rgb,
            "source_depth": source_depth,
            "size": entry.get("size"),
            "source_size": entry.get("source_size"),
            "category": entry.get("category"),
            "frame": entry.get("frame"),
            "rel_path": entry.get("rel_path"),
            "metadata": entry,
        }

    def _manifest_entry_in_split(
        self,
        entry: Dict[str, Any],
        whitelist: set[str],
    ) -> bool:
        if not whitelist:
            return True

        candidates: set[str] = set()
        rel_path = entry.get("rel_path")
        if rel_path:
            rel_norm = rel_path.replace("\\", "/")
            candidates.add(rel_norm)
            candidates.add(rel_norm.replace("/", os.sep))

        category = entry.get("category")
        frame = entry.get("frame")
        if category and frame:
            candidates.add(f"{category}/{frame}")
            candidates.add(f"{category}\\{frame}")

        image_path = entry.get("ref") or entry.get("image")
        if image_path:
            resolved = self._resolve_path(image_path)
            if resolved and resolved.startswith(self.data_root):
                rel_candidate = os.path.relpath(resolved, self.data_root)
                candidates.update({rel_candidate, rel_candidate.replace(os.sep, "/")})

        return any(candidate in whitelist for candidate in candidates)

    def _read_manifest_split_whitelist(self) -> Optional[set[str]]:
        if self._manifest_split_whitelist is not None:
            return self._manifest_split_whitelist or None

        whitelist: set[str] = set()
        candidates = [
            self.manifest_split_path,
            self.split_file if isinstance(self.split_file, str) else None,
        ]
        for candidate in candidates:
            candidate_path = self._resolve_path(candidate) if candidate else None
            if not candidate_path or not os.path.isfile(candidate_path):
                continue
            with open(candidate_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    token = line.strip()
                    if not token:
                        continue
                    whitelist.add(token.split()[0])

        self._manifest_split_whitelist = whitelist or None
        return self._manifest_split_whitelist

    def _resolve_path(self, path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        expanded = os.path.expanduser(path)
        if not os.path.isabs(expanded):
            expanded = os.path.join(self.data_root, expanded)
        return os.path.normpath(expanded)

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [True] * self.num_frames * self.num_copies
        results["quality"] = [1] * self.num_frames * self.num_copies
        return results

    def get_mapper(self):
        return {
            "image_filename": 0,
            "depth_filename": 1,
            "K": 2,
        }

    def get_single_item(self, idx, sample=None, mapper=None):
        if getattr(self, "_use_manifest", False):
            return self._get_single_item_from_manifest(idx)
        return super().get_single_item(idx, sample=sample, mapper=mapper)

    def _get_single_item_from_manifest(self, idx: int):
        sample = self.dataset[idx]
        results = {
            (0, 0): dict(
                gt_fields=set(),
                image_fields=set(),
                mask_fields=set(),
                camera_fields=set(),
            )
        }
        results = self.pre_pipeline(results)
        results["sequence_fields"] = [(0, 0)]
        seq = results[(0, 0)]

        image_path = sample["image_path"]
        image_tensor = read_image(image_path)
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)
        seq["filename"] = image_path
        seq["image_fields"].add("image")
        seq["image_ori_shape"] = image_tensor.shape[-2:]
        seq["image"] = image_tensor.unsqueeze(0)

        stack_paths = sample.get("stack_paths", [])
        if stack_paths:
            stack_tensors: List[torch.Tensor] = []
            for stack_path in stack_paths:
                stack_tensor = read_image(stack_path)
                if stack_tensor.shape[0] == 1:
                    stack_tensor = stack_tensor.repeat(3, 1, 1)
                if stack_tensor.shape[-2:] != image_tensor.shape[-2:]:
                    with Image.open(stack_path) as stack_img:
                        stack_img = stack_img.convert("RGB").resize(
                            (image_tensor.shape[-1], image_tensor.shape[-2]),
                            Image.BICUBIC,
                        )
                        stack_tensor = torch.from_numpy(
                            np.asarray(stack_img, dtype=np.uint8)
                        ).permute(2, 0, 1)
                stack_tensors.append(stack_tensor)
            if stack_tensors:
                seq["defocus_stack"] = torch.stack(stack_tensors, dim=0)
                seq["image_fields"].add("defocus_stack")
                k_values = sample.get("k_values") or []
                seq["k_values"] = (
                    torch.tensor(k_values, dtype=torch.float32) if k_values else None
                )

        depth_array = self._load_manifest_depth(sample)
        mask_array = self._load_manifest_mask(depth_array.shape)

        finite_mask = np.isfinite(depth_array)
        positive_mask = depth_array > self.min_depth
        far_mask = depth_array < self.max_depth
        valid_mask = finite_mask & positive_mask & far_mask & (mask_array > 0.5)
        if not np.count_nonzero(valid_mask):
            raise ValueError(f"Make3D depth map {sample['depth_path']} has no valid pixels.")

        depth_array = np.where(valid_mask, depth_array, 0.0).astype(np.float32, copy=False)
        depth_tensor = torch.from_numpy(depth_array).view(1, 1, *depth_array.shape)

        seq["gt_fields"].add("depth")
        seq["depth_ori_shape"] = depth_array.shape
        seq["depth"] = depth_tensor

        validity_tensor = torch.from_numpy(valid_mask.astype(np.float32)).view(1, 1, *depth_array.shape)
        seq["mask_fields"].update({"validity_mask", "depth_mask"})
        seq["validity_mask"] = validity_tensor
        seq["depth_mask"] = validity_tensor.clone()

        seq["metadata"] = sample.get("metadata", {}).copy()
        seq["metadata"]["eval_protocol"] = self.eval_protocol
        if sample.get("stack_index"):
            seq["metadata"]["stack_index_path"] = sample["stack_index"]
        if sample.get("source_depth"):
            seq["metadata"]["source_depth_path"] = sample["source_depth"]
        if sample.get("source_rgb"):
            seq["metadata"]["source_rgb_path"] = sample["source_rgb"]

        seq["camera_fields"].update({"camera", "cam2w"})
        K = self._build_intrinsics(image_tensor)
        camera = BatchCamera.from_camera(Pinhole(K=K.unsqueeze(0)))
        seq["camera"] = camera
        seq["cam2w"] = torch.eye(4, dtype=K.dtype).unsqueeze(0)

        if self.debug_augmentation and "defocus_stack" in seq:
            print(
                f"[Make3D] Input image {image_tensor.shape}, defocus stack {seq['defocus_stack'].shape}"
            )

        results = self.preprocess(results)
        if not self.test_mode:
            results = self.augment(results)
        results = self.postprocess(results)

        if self.debug_augmentation:
            seq0 = results.get("seq0", {})
            if isinstance(seq0, dict):
                image_shape = seq0.get("image", torch.empty(0)).shape
                stack_shape = seq0.get("defocus_stack", torch.empty(0)).shape
                print(
                    f"[Make3D] Output image {image_shape}, defocus stack {stack_shape}"
                )

        return results

    def _load_manifest_depth(self, sample: Dict[str, Any]) -> np.ndarray:
        if loadmat is None:
            raise ImportError("scipy is required to load Make3D .mat depth files.")
        depth_path = sample["depth_path"]
        data = loadmat(depth_path)
        if "Position3DGrid" not in data:
            raise KeyError(f"Position3DGrid not found in {depth_path}")
        grid = data["Position3DGrid"]
        if grid.ndim != 3 or grid.shape[2] <= self.depth_channel:
            raise ValueError(
                f"Unexpected Position3DGrid shape {grid.shape} (channel={self.depth_channel})"
            )
        depth = grid[:, :, self.depth_channel].astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        if self.max_depth_clip is not None:
            depth = np.clip(depth, 0.0, self.max_depth_clip)
        return depth

    def _load_manifest_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        return np.ones(shape, dtype=np.float32)

    def _build_intrinsics(self, image_tensor: torch.Tensor) -> torch.Tensor:
        height, width = image_tensor.shape[-2:]
        fx = 0.7 * width
        fy = fx
        cx = 0.5 * width
        cy = 0.5 * height
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        return K

    def get_evaluation(self, metrics=None):
        results = super().get_evaluation(metrics)
        suffix = self.eval_protocol.lower()
        renamed: Dict[str, float] = {}
        for key, value in results.items():
            renamed[f"{key}_{suffix}"] = value
        return renamed

    def _size_to_hw(self, size: Optional[Iterable[int]]) -> Optional[Tuple[int, int]]:
        if not size:
            return None
        size = list(size)
        if len(size) < 2:
            return None
        width, height = size[0], size[1]
        return int(height), int(width)
