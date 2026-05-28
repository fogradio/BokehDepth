import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from torchvision.io import read_image

from unidepth.datasets.image_dataset import ImageDataset
from unidepth.datasets.utils import DatasetFromList
from unidepth.utils.camera import BatchCamera, Pinhole


class HAMMER(ImageDataset):
    """HAMMER dataset supporting both legacy HDF5 and manifest-based defocus stacks."""

    min_depth = 0.01
    max_depth = 10.0
    depth_scale = 1000.0
    train_split = "test.txt"
    test_split = "test.txt"
    intrisics_file = "intrinsics.json"
    hdf5_paths = ["hammer.hdf5"]

    def __init__(
        self,
        image_shape,
        split_file,
        test_mode,
        crop=None,
        benchmark=False,
        augmentations_db=None,
        normalize: bool = True,
        resize_method: str = "hard",
        mini: float = 1.0,
        manifest_path: str | None = None,
        manifest_split_path: str | None = None,
        defocus_stack_indices=None,
        debug_augmentation: bool = False,
        **kwargs,
    ):
        augmentations_db = augmentations_db or {}

        manifest_path = manifest_path or kwargs.pop("manifest_path", None)
        manifest_split_path = manifest_split_path or kwargs.pop("manifest_split_path", None)
        defocus_stack_indices = (
            defocus_stack_indices if defocus_stack_indices is not None else kwargs.pop("defocus_stack_indices", None)
        )
        debug_augmentation = debug_augmentation or kwargs.pop("debug_augmentation", False)

        self.manifest_path = manifest_path or os.environ.get("HAMMER_MANIFEST_PATH")
        self.manifest_split_path = manifest_split_path
        self.debug_augmentation = bool(debug_augmentation)

        if defocus_stack_indices is not None:
            if isinstance(defocus_stack_indices, (str, bytes)):
                indices = str(defocus_stack_indices).split(",")
                defocus_stack_indices = [int(idx) for idx in indices if idx]
            self.defocus_stack_indices = [int(i) for i in defocus_stack_indices]
        else:
            self.defocus_stack_indices = None

        self._use_manifest = False
        self._manifest_split_whitelist: Optional[set[str]] = None
        self._intrinsics_cache: Dict[str, torch.Tensor] = {}

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
        self.crop = crop
        self.load_dataset()

    def load_dataset(self) -> None:
        manifest_path = self.manifest_path
        if manifest_path:
            manifest_path = self._resolve_path(manifest_path)
            if not manifest_path or not os.path.isfile(manifest_path):
                raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
            self._load_manifest_dataset(manifest_path)
            return

        h5file = h5py.File(
            os.path.join(self.data_root, self.hdf5_paths[0]),
            "r",
            libver="latest",
            swmr=True,
        )
        txt_file = np.array(h5file[self.split_file])
        txt_string = txt_file.tobytes().decode("ascii")[:-1]  # correct the -1
        intrinsics = np.array(h5file[self.intrisics_file]).tobytes().decode("ascii")
        intrinsics = json.loads(intrinsics)
        h5file.close()
        dataset = []
        for line in txt_string.split("\n"):
            image_filename, depth_filename = line.strip().split(" ")
            intrinsics_val = torch.tensor(intrinsics[image_filename]).squeeze()[:, :3]
            sample = [image_filename, depth_filename, intrinsics_val]
            dataset.append(sample)

        self.dataset = DatasetFromList(dataset)
        self.log_load_dataset()

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
                f"Parsed manifest {manifest_path} but no valid HAMMER samples were found."
            )

        if not self.test_mode:
            records = self.chunk(records, chunk_dim=1, pct=self.mini)

        self.dataset = DatasetFromList(records, serialize=False)
        self._use_manifest = True
        # Manifest depth is already expressed in metres.
        self.depth_scale = 1.0
        self.manifest_path = manifest_path
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
            stack_resolved = self._resolve_path(stack_item)
            if stack_resolved and os.path.isfile(stack_resolved):
                stack_paths.append(stack_resolved)

        k_values = entry.get("k", [])
        if self.defocus_stack_indices and stack_paths:
            selected_paths: List[str] = []
            selected_k: List[float] = []
            for idx in self.defocus_stack_indices:
                if 0 <= idx < len(stack_paths):
                    selected_paths.append(stack_paths[idx])
                    if k_values and idx < len(k_values):
                        selected_k.append(k_values[idx])
            if selected_paths:
                stack_paths = selected_paths
                if k_values:
                    k_values = selected_k if selected_k else []

        mask_paths: List[str] = []
        for key in ("mask", "masks", "mask_list", "image_mask", "source_mask"):
            value = entry.get(key)
            if not value:
                continue
            candidates = value if isinstance(value, (list, tuple)) else [value]
            for candidate in candidates:
                resolved = self._resolve_path(candidate)
                if resolved and os.path.isfile(resolved):
                    mask_paths.append(resolved)

        stack_index = self._resolve_path(entry.get("stack_index"))
        intrinsics_path = self._resolve_path(entry.get("source_intrinsics") or entry.get("intrinsics"))

        intrinsics_matrix = entry.get("intrinsics_matrix") or entry.get("camera_intrinsics")
        if intrinsics_matrix is not None:
            intrinsics_matrix = np.asarray(intrinsics_matrix, dtype=np.float32).reshape(3, 3)

        source_depth = self._resolve_path(entry.get("source_depth"))
        source_rgb = self._resolve_path(entry.get("source_rgb") or entry.get("ref"))
        source_mask = self._resolve_path(entry.get("source_mask"))

        return {
            "image_path": image_path,
            "depth_path": depth_path,
            "stack_paths": stack_paths,
            "k_values": k_values,
            "mask_paths": mask_paths,
            "stack_index": stack_index,
            "intrinsics_path": intrinsics_path,
            "intrinsics_matrix": intrinsics_matrix,
            "source_depth": source_depth,
            "source_rgb": source_rgb,
            "source_mask": source_mask,
            "size": entry.get("size"),
            "source_size": entry.get("source_size"),
            "scene": entry.get("scene"),
            "sensor": entry.get("sensor"),
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
                    token = token.split()[0]
                    whitelist.add(token)

        self._manifest_split_whitelist = whitelist or None
        return self._manifest_split_whitelist

    def _resolve_path(self, path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        path_exp = os.path.expanduser(path)
        if not os.path.isabs(path_exp):
            path_exp = os.path.join(self.data_root, path_exp)
        return os.path.normpath(path_exp)

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
                    torch.tensor(k_values, dtype=torch.float32) if k_values else None
                )

        depth_array = self._load_manifest_depth(sample)
        mask_array = self._load_manifest_mask(
            sample.get("mask_paths") or [],
            depth_array.shape[-2:],
        )

        finite_mask = np.isfinite(depth_array)
        positive_mask = depth_array > self.min_depth
        far_mask = depth_array < self.max_depth
        valid_mask = finite_mask & positive_mask & far_mask & (mask_array > 0.5)
        if not np.count_nonzero(valid_mask):
            raise ValueError(f"HAMMER depth map {sample['depth_path']} has no valid pixels.")

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
        if sample.get("stack_index"):
            seq["metadata"]["stack_index_path"] = sample["stack_index"]
        if sample.get("source_depth"):
            seq["metadata"]["source_depth_path"] = sample["source_depth"]
        if sample.get("source_rgb"):
            seq["metadata"]["source_rgb_path"] = sample["source_rgb"]
        if sample.get("source_mask"):
            seq["metadata"]["source_mask_path"] = sample["source_mask"]
        if sample.get("k_values"):
            seq["metadata"]["defocus_k_values"] = sample["k_values"]

        seq["camera_fields"].update({"camera", "cam2w"})
        K = self._get_manifest_intrinsics(sample)
        if K is None:
            import warnings

            fallback_fx = 0.7 * image_tensor.shape[-1]
            fallback_cx = 0.5 * image_tensor.shape[-1]
            fallback_cy = 0.5 * image_tensor.shape[-2]
            warnings.warn(
                f"[HAMMER] Missing intrinsics for {sample.get('image_path', 'unknown')}. "
                f"Using heuristic fallback intrinsics: fx=fy={fallback_fx:.2f}, cx={fallback_cx:.2f}, cy={fallback_cy:.2f}.",
                UserWarning,
                stacklevel=2,
            )
            K = torch.eye(3, dtype=torch.float32)
            K[0, 0] = K[1, 1] = fallback_fx
            K[0, 2] = fallback_cx
            K[1, 2] = fallback_cy

        camera = BatchCamera.from_camera(Pinhole(K=K.unsqueeze(0)))
        seq["camera"] = camera
        seq["cam2w"] = torch.eye(4, dtype=K.dtype).unsqueeze(0)

        if self.debug_augmentation and "defocus_stack" in seq:
            print(
                f"[HAMMER] Input image shape: {seq['image'].shape}, "
                f"defocus stack: {seq['defocus_stack'].shape}"
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
                    f"[HAMMER] Output image shape: {image_shape}, "
                    f"defocus stack: {stack_shape}"
                )

        return results

    def _load_manifest_depth(self, sample: Dict[str, Any]) -> np.ndarray:
        depth_path = sample["depth_path"]
        with Image.open(depth_path) as depth_img:
            depth = np.asarray(depth_img, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth *= 1.0 / 1000.0  # millimetre -> metre
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def _load_manifest_mask(
        self,
        mask_paths: Sequence[str],
        shape: Tuple[int, int],
    ) -> np.ndarray:
        if not mask_paths:
            return np.ones(shape, dtype=np.float32)

        combined = np.ones(shape, dtype=np.float32)
        for mask_path in mask_paths:
            try:
                with Image.open(mask_path) as mask_img:
                    mask_arr = np.array(mask_img)
            except FileNotFoundError:
                continue
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[..., 0]
            if mask_arr.shape != shape:
                mask_arr = np.array(
                    Image.fromarray(mask_arr).resize(
                        (shape[1], shape[0]), resample=Image.NEAREST
                    )
                )
            current_valid = (mask_arr == 0).astype(np.float32)
            combined *= current_valid
        return combined

    def _size_to_hw(self, size: Optional[Iterable[int]]) -> Optional[Tuple[int, int]]:
        if not size:
            return None
        size = list(size)
        if len(size) < 2:
            return None
        width, height = size[0], size[1]
        return int(height), int(width)

    def _get_manifest_intrinsics(self, sample: Dict[str, Any]) -> Optional[torch.Tensor]:
        intrinsics_path = sample.get("intrinsics_path")
        if intrinsics_path:
            K = self._intrinsics_cache.get(intrinsics_path)
            if K is None and os.path.isfile(intrinsics_path):
                matrix = self._load_intrinsics_file(intrinsics_path)
                if matrix is not None:
                    K = torch.from_numpy(matrix.astype(np.float32))
                    self._intrinsics_cache[intrinsics_path] = K
            if K is not None:
                return K.clone()

        matrix = sample.get("intrinsics_matrix")
        if matrix is not None:
            return torch.from_numpy(matrix.astype(np.float32)).clone()

        return None

    def _load_intrinsics_file(self, path: str) -> Optional[np.ndarray]:
        try:
            matrix = np.loadtxt(path, dtype=np.float32)
        except OSError:
            return None
        if matrix.size == 9:
            matrix = matrix.reshape(3, 3)
        return matrix.astype(np.float32)
