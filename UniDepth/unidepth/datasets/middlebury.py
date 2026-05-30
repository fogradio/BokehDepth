"""Middlebury Stereo 2014 manifest loader with defocus-stack support."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.io import read_image

from unidepth.datasets.image_dataset import ImageDataset
from unidepth.datasets.utils import DatasetFromList
from unidepth.utils.camera import BatchCamera, Pinhole


_PFM_FLOAT_HEADERS = {"Pf", "PF"}


class Middlebury2014(ImageDataset):
    """Loads Middlebury 2014 Perfect scenes via manifest JSONL.

    Each manifest entry must at least provide the reference RGB path (``ref``).
    The depth file is inferred as ``disp0.pfm`` next to ``im0.png`` unless a
    custom ``depth`` path is supplied. Calibration ``calib.txt`` is resolved via
    ``source_intrinsics`` or derived from ``im0.png``.
    """

    min_depth = 0.01
    max_depth = 80.0
    depth_scale = 1.0
    test_split = "middlebury.txt"
    train_split = "middlebury.txt"

    def __init__(
        self,
        image_shape,
        split_file,
        test_mode,
        augmentations_db: Dict[str, Any] | None = None,
        normalize: bool = True,
        resize_method: str = "hard",
        mini: float = 1.0,
        benchmark: bool = False,
        **kwargs,
    ) -> None:
        augmentations_db = augmentations_db or {}

        manifest_path = kwargs.pop("manifest_path", None)
        manifest_split_path = kwargs.pop("manifest_split_path", None)
        defocus_stack_indices = kwargs.pop("defocus_stack_indices", None)

        self.manifest_path = manifest_path or os.environ.get("MIDDLEBURY_MANIFEST_PATH")
        self.manifest_split_path = manifest_split_path
        if defocus_stack_indices is not None:
            if isinstance(defocus_stack_indices, (str, bytes)):
                tokens = str(defocus_stack_indices).split(",")
                defocus_stack_indices = [int(tok) for tok in tokens if tok]
            self.defocus_stack_indices = [int(idx) for idx in defocus_stack_indices]
        else:
            self.defocus_stack_indices = None

        self._use_manifest = False
        self._manifest_split_whitelist: Optional[set[str]] = None
        self._intrinsics_cache: Dict[str, Tuple[torch.Tensor, float]] = {}

        super().__init__(
            image_shape=image_shape,
            split_file=split_file,
            test_mode=test_mode,
            benchmark=benchmark,
            normalize=normalize,
            augmentations_db=augmentations_db,
            resize_method=resize_method,
            mini=mini,
            **kwargs,
        )
        self.test_mode = test_mode
        self.load_dataset()

    # ---------------------------------------------------------------------
    # Dataset loading
    # ---------------------------------------------------------------------
    def load_dataset(self) -> None:
        manifest_path = self.manifest_path
        if not manifest_path:
            raise ValueError("Middlebury2014 dataset requires --manifest-path or MIDDLEBURY_MANIFEST_PATH")
        manifest_path = self._resolve_path(manifest_path)
        if not manifest_path or not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        self._load_manifest_dataset(manifest_path)

    def _load_manifest_dataset(self, manifest_path: str) -> None:
        records: List[Dict[str, Any]] = []
        split_whitelist = self._read_manifest_split_whitelist()

        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line in handle:
                entry_text = line.strip()
                if not entry_text:
                    continue
                try:
                    entry = json.loads(entry_text)
                except json.JSONDecodeError:
                    continue

                if split_whitelist and not self._manifest_entry_in_split(entry, split_whitelist):
                    continue

                record = self._build_manifest_sample(entry)
                if record:
                    records.append(record)

        if not records:
            raise FileNotFoundError(
                f"Parsed manifest {manifest_path} but no valid Middlebury samples were found."
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
        if not image_path:
            return None
        image_path = self._resolve_path(image_path)
        if not image_path or not os.path.isfile(image_path):
            return None

        depth_path = entry.get("depth")
        if depth_path:
            depth_path = self._resolve_path(depth_path)
        else:
            depth_path = self._derive_disp_path(image_path)
        if not depth_path or not os.path.isfile(depth_path):
            return None

        calib_path = entry.get("source_intrinsics")
        if calib_path:
            calib_path = self._resolve_path(calib_path)
        else:
            calib_path = self._guess_calib_from_image(image_path)
        if not calib_path or not os.path.isfile(calib_path):
            return None

        stack_paths: List[str] = []
        for stack_item in entry.get("stack", []):
            resolved = self._resolve_path(stack_item)
            if resolved and os.path.isfile(resolved):
                stack_paths.append(resolved)

        k_values = entry.get("k", [])
        if self.defocus_stack_indices and stack_paths:
            sel_paths: List[str] = []
            sel_k: List[float] = []
            for idx in self.defocus_stack_indices:
                if 0 <= idx < len(stack_paths):
                    sel_paths.append(stack_paths[idx])
                    if k_values and idx < len(k_values):
                        sel_k.append(k_values[idx])
            if sel_paths:
                stack_paths = sel_paths
                if k_values:
                    k_values = sel_k

        stack_index = entry.get("stack_index")
        if stack_index:
            stack_index = self._resolve_path(stack_index)

        metadata = {
            "scene": entry.get("scene"),
            "frame": entry.get("frame"),
            "rel_path": entry.get("rel_path"),
            "size": entry.get("size"),
            "source_size": entry.get("source_size"),
        }
        if stack_index:
            metadata["stack_index"] = stack_index
        if entry.get("stack"):
            metadata["stack_raw"] = entry.get("stack")

        return {
            "image_path": image_path,
            "depth_path": depth_path,
            "calib_path": calib_path,
            "stack_paths": stack_paths,
            "k_values": k_values,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------
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
        scene = entry.get("scene")
        frame = entry.get("frame")
        if scene and frame:
            candidates.add(f"{scene}/{frame}")
            candidates.add(f"{scene}\\{frame}")
        image_path = entry.get("ref") or entry.get("image")
        if image_path:
            resolved = self._resolve_path(image_path)
            if resolved:
                candidates.add(os.path.relpath(resolved, self.data_root))
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
        path_exp = os.path.expanduser(path)
        if not os.path.isabs(path_exp):
            path_exp = os.path.join(self.data_root, path_exp)
        return os.path.normpath(path_exp)

    # ------------------------------------------------------------------
    # Dataloader plumbing
    # ------------------------------------------------------------------
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
                    with Image.open(stack_path) as img:
                        img = img.convert("RGB")
                        img = img.resize(
                            (image_tensor.shape[-1], image_tensor.shape[-2]),
                            Image.BICUBIC,
                        )
                        stack_tensor = torch.from_numpy(
                            np.asarray(img, dtype=np.uint8)
                        ).permute(2, 0, 1)
                stack_tensors.append(stack_tensor)
            if stack_tensors:
                seq["defocus_stack"] = torch.stack(stack_tensors, dim=0)
                seq["image_fields"].add("defocus_stack")
                k_values = sample.get("k_values") or []
                seq["k_values"] = (
                    torch.tensor(k_values, dtype=torch.float32)
                    if k_values
                    else None
                )

        depth_array, mask_array = self._load_manifest_depth(sample)
        if mask_array.dtype != np.bool_:
            mask_array = mask_array.astype(np.bool_)
        finite_mask = np.isfinite(depth_array)
        positive_mask = depth_array > self.min_depth
        far_mask = depth_array < self.max_depth
        valid_mask = finite_mask & positive_mask & far_mask & mask_array
        if not np.count_nonzero(valid_mask):
            raise ValueError(f"Middlebury depth map {sample['depth_path']} has no valid pixels.")

        depth_array = np.where(valid_mask, depth_array, 0.0).astype(np.float32, copy=False)
        depth_tensor = torch.from_numpy(depth_array).view(1, 1, *depth_array.shape)

        seq["gt_fields"].add("depth")
        seq["depth_ori_shape"] = depth_array.shape
        seq["depth"] = depth_tensor

        validity_tensor = torch.from_numpy(valid_mask.astype(np.float32)).view(
            1, 1, *depth_array.shape
        )
        seq["mask_fields"].update({"validity_mask", "depth_mask"})
        seq["validity_mask"] = validity_tensor
        seq["depth_mask"] = validity_tensor.clone()

        seq["metadata"] = sample.get("metadata", {}).copy()
        seq["metadata"]["depth_path"] = sample["depth_path"]
        seq["metadata"]["calib_path"] = sample["calib_path"]

        K, baseline_m = self._get_intrinsics(sample["calib_path"])
        seq["metadata"]["baseline_m"] = baseline_m

        seq["camera_fields"].update({"camera", "cam2w"})
        camera = BatchCamera.from_camera(Pinhole(K=K.unsqueeze(0)))
        seq["camera"] = camera
        seq["cam2w"] = torch.eye(4, dtype=K.dtype).unsqueeze(0)

        results = self.preprocess(results)
        if not self.test_mode:
            results = self.augment(results)
        results = self.postprocess(results)

        return results

    # ------------------------------------------------------------------
    # Depth utilities
    # ------------------------------------------------------------------
    def _load_manifest_depth(self, sample: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        depth_path = sample["depth_path"]
        disparity = self._read_pfm(depth_path)
        K, baseline_m = self._get_intrinsics(sample["calib_path"])
        focal_px = float(K[0, 0])
        depth = np.zeros_like(disparity, dtype=np.float32)
        mask = disparity > 0
        depth[mask] = (focal_px * baseline_m) / (disparity[mask] + 1e-6)
        return depth, mask

    def _read_pfm(self, path: str) -> np.ndarray:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"PFM file not found: {path}")
        with open(path, "rb") as handle:
            header = handle.readline().decode("utf-8").strip()
            if header not in _PFM_FLOAT_HEADERS:
                raise ValueError(f"Invalid PFM header '{header}' in {path}")

            dims_line = handle.readline().decode("utf-8").strip()
            while dims_line.startswith("#"):
                dims_line = handle.readline().decode("utf-8").strip()
            width, height = map(int, dims_line.split())

            scale = float(handle.readline().decode("utf-8").strip())
            endian = "<" if scale < 0 else ">"
            data = np.fromfile(handle, endian + "f", width * height)
        if data.size != width * height:
            raise ValueError(f"Unexpected payload size in {path}")
        data = np.flipud(data.reshape((height, width)))
        return data.astype(np.float32)

    def _get_intrinsics(self, calib_path: str) -> Tuple[torch.Tensor, float]:
        calib_norm = os.path.normpath(calib_path)
        cached = self._intrinsics_cache.get(calib_norm)
        if cached is not None:
            K_cached, baseline_cached = cached
            return K_cached.clone(), float(baseline_cached)

        cam0, baseline_mm = self._parse_calibration(calib_norm)
        K = torch.from_numpy(cam0.astype(np.float32))
        baseline_m = baseline_mm / 1000.0
        self._intrinsics_cache[calib_norm] = (K, baseline_m)
        return K.clone(), baseline_m

    def _parse_calibration(self, calib_path: str) -> Tuple[np.ndarray, float]:
        if not os.path.isfile(calib_path):
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")
        cam0: Optional[np.ndarray] = None
        baseline_mm: Optional[float] = None
        with open(calib_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line.startswith("cam0="):
                    content = line.split("=", 1)[1].strip().strip("[]")
                    values = [float(v) for v in re.split(r"[;\s]+", content) if v]
                    if len(values) != 9:
                        raise ValueError(f"Invalid cam0 matrix in {calib_path}")
                    cam0 = np.array(values, dtype=np.float32).reshape(3, 3)
                elif line.startswith("baseline="):
                    baseline_mm = float(line.split("=", 1)[1].strip())
        if cam0 is None or baseline_mm is None:
            raise ValueError(f"Missing cam0/baseline in {calib_path}")
        return cam0, baseline_mm

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def _derive_disp_path(self, image_path: str) -> str:
        dirname, filename = os.path.split(image_path)
        if filename.lower() != "im0.png":
            return os.path.join(dirname, "disp0.pfm")
        return os.path.join(dirname, "disp0.pfm")

    def _guess_calib_from_image(self, image_path: str) -> str:
        dirname = os.path.dirname(image_path)
        return os.path.join(dirname, "calib.txt")

    def _size_to_hw(self, size: Optional[Iterable[int]]) -> Optional[Tuple[int, int]]:
        if not size:
            return None
        size = list(size)
        if len(size) < 2:
            return None
        width, height = size[0], size[1]
        return int(height), int(width)
