import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from torchvision.io import read_image

from unidepth.datasets.image_dataset import ImageDataset
from unidepth.datasets.sequence_dataset import SequenceDataset
from unidepth.datasets.utils import DatasetFromList
from unidepth.utils.camera import BatchCamera, Pinhole


class ETH3D(ImageDataset):
    min_depth = 0.01
    max_depth = 50.0
    depth_scale = 1000.0
    test_split = "train.txt"
    train_split = "train.txt"
    intrisics_file = "intrinsics.json"
    hdf5_paths = ["ETH3D.hdf5"]

    def __init__(
        self,
        image_shape,
        split_file,
        test_mode,
        benchmark: bool = False,
        augmentations_db: Dict[str, Any] | None = None,
        normalize: bool = True,
        resize_method: str = "hard",
        mini: float = 1.0,
        **kwargs,
    ) -> None:
        augmentations_db = augmentations_db or {}

        manifest_path = kwargs.pop("manifest_path", None)
        manifest_split_path = kwargs.pop("manifest_split_path", None)
        defocus_stack_indices = kwargs.pop("defocus_stack_indices", None)
        debug_augmentation = kwargs.pop("debug_augmentation", False)

        self.manifest_path = manifest_path or os.environ.get("ETH3D_MANIFEST_PATH")
        self.manifest_split_path = manifest_split_path
        if defocus_stack_indices is not None:
            if isinstance(defocus_stack_indices, (str, bytes)):
                indices = str(defocus_stack_indices).split(",")
                defocus_stack_indices = [int(idx) for idx in indices if idx]
            self.defocus_stack_indices = [int(i) for i in defocus_stack_indices]
        else:
            self.defocus_stack_indices = None

        self.debug_augmentation = bool(debug_augmentation)
        self._use_manifest = False
        self._manifest_split_whitelist: Optional[set[str]] = None
        self._manifest_intrinsics_cache: Dict[str, torch.Tensor] = {}
        self._scene_intrinsics_cache: Dict[str, Dict[str, torch.Tensor]] = {}

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
        txt_string = txt_file.tostring().decode("ascii")[:-1]  # correct the -1
        intrinsics = np.array(h5file[self.intrisics_file]).tostring().decode("ascii")
        intrinsics = json.loads(intrinsics)
        h5file.close()

        dataset: List[List[Any]] = []
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
                f"Parsed manifest {manifest_path} but no valid ETH3D samples were found."
            )

        if not self.test_mode:
            records = self.chunk(records, chunk_dim=1, pct=self.mini)

        self.dataset = DatasetFromList(records, serialize=False)
        self._use_manifest = True
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
            if isinstance(value, (list, tuple)):
                candidates = value
            else:
                candidates = [value]
            for candidate in candidates:
                resolved = self._resolve_path(candidate)
                if resolved and os.path.isfile(resolved):
                    mask_paths.append(resolved)

        stack_index = entry.get("stack_index")
        if stack_index:
            stack_index = self._resolve_path(stack_index)

        source_rgb = entry.get("source_rgb")
        if source_rgb:
            source_rgb = self._resolve_path(source_rgb)
        source_depth = entry.get("source_depth")
        if source_depth:
            source_depth = self._resolve_path(source_depth)
        source_mask = entry.get("source_mask")
        if source_mask:
            source_mask = self._resolve_path(source_mask)

        return {
            "image_path": image_path,
            "depth_path": depth_path,
            "stack_paths": stack_paths,
            "k_values": k_values,
            "mask_paths": mask_paths,
            "stack_index": stack_index,
            "source_rgb": source_rgb,
            "source_depth": source_depth,
            "source_mask": source_mask,
            "size": entry.get("size"),
            "source_size": entry.get("source_size"),
            "scene": entry.get("scene"),
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
                    torch.tensor(k_values, dtype=torch.float32)
                    if k_values
                    else None
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
            raise ValueError(f"ETH3D depth map {sample['depth_path']} has no valid pixels.")

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

        seq["camera_fields"].update({"camera", "cam2w"})
        K = self._get_manifest_intrinsics(sample)
        if K is None:
            # Warning: no COLMAP calibrated intrinsics; falling back to a heuristic.
            import warnings
            fallback_fx = 0.7 * image_tensor.shape[-1]
            fallback_cx = 0.5 * image_tensor.shape[-1]
            fallback_cy = 0.5 * image_tensor.shape[-2]
            warnings.warn(
                f"[ETH3D] no COLMAP camera intrinsics found for image {sample.get('image_path', 'unknown')}. "
                f"Using heuristic fallback intrinsics: fx=fy={fallback_fx:.2f}, cx={fallback_cx:.2f}, cy={fallback_cy:.2f}. "
                f"This may degrade depth-estimation accuracy; check whether the dslr_calibration_jpg/ folder exists under scene directory {sample.get('scene', 'unknown')}.",
                UserWarning,
                stacklevel=2
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
                f"[ETH3D] Input image shape: {seq['image'].shape}, "
                f"defocus stack: {seq['defocus_stack'].shape}"
            )

        results = self.preprocess(results)
        if not self.test_mode:
            results = self.augment(results)
        results = self.postprocess(results)

        if self.debug_augmentation and "seq0" in results:
            seq0 = results.get("seq0", {})
            if isinstance(seq0, dict):
                image_shape = seq0.get("image", torch.empty(0)).shape
                stack_shape = seq0.get("defocus_stack", torch.empty(0)).shape
                print(
                    f"[ETH3D] Output image shape: {image_shape}, "
                    f"defocus stack: {stack_shape}"
                )

        return results

    def _load_manifest_depth(self, sample: Dict[str, Any]) -> np.ndarray:
        depth_path = sample["depth_path"]
        source_shape = self._size_to_hw(sample.get("source_size"))
        fallback_shape = self._size_to_hw(sample.get("size"))

        ext = os.path.splitext(depth_path)[1].lower()
        depth: np.ndarray
        if ext == ".npy":
            depth = np.load(depth_path)
        elif ext == ".npz":
            data = np.load(depth_path)
            if "depth" in data:
                depth = data["depth"]
            else:
                first_key = list(data.keys())[0]
                depth = data[first_key]
        elif ext in {".png", ".tif", ".tiff"}:
            depth = np.asarray(Image.open(depth_path), dtype=np.float32)
        else:
            array = np.fromfile(depth_path, dtype="<f4")
            target_shape = source_shape or fallback_shape
            if target_shape is None:
                raise ValueError(
                    f"Cannot infer depth shape for ETH3D manifest sample: {depth_path}"
                )
            expected = target_shape[0] * target_shape[1]
            if array.size != expected:
                raise ValueError(
                    f"Depth size mismatch for {depth_path}: expected {expected}, got {array.size}"
                )
            depth = array.reshape(target_shape)

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth.squeeze()
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
        image_path = os.path.normpath(sample["image_path"])
        cached = self._manifest_intrinsics_cache.get(image_path)
        if cached is not None:
            return cached.clone()

        scene_dir = self._infer_scene_dir(image_path)
        if not scene_dir:
            return None

        if scene_dir not in self._scene_intrinsics_cache:
            self._scene_intrinsics_cache[scene_dir] = self._parse_scene_intrinsics(scene_dir)

        scene_intrinsics = self._scene_intrinsics_cache.get(scene_dir, {})
        K = scene_intrinsics.get(image_path)
        if K is not None:
            self._manifest_intrinsics_cache[image_path] = K
            return K.clone()
        return None

    def _infer_scene_dir(self, image_path: str) -> Optional[str]:
        scene_dir = os.path.dirname(os.path.dirname(os.path.dirname(image_path)))
        if os.path.isdir(scene_dir):
            return scene_dir

        try:
            rel_parts = os.path.relpath(image_path, self.data_root).split(os.sep)
        except ValueError:
            return None
        if len(rel_parts) < 2:
            return None
        candidate = os.path.join(self.data_root, rel_parts[0], rel_parts[1])
        if os.path.isdir(candidate):
            return candidate
        return None

    def _parse_scene_intrinsics(self, scene_dir: str) -> Dict[str, torch.Tensor]:
        calibration_dir = os.path.join(scene_dir, "dslr_calibration_jpg")
        cameras_file = os.path.join(calibration_dir, "cameras.txt")
        images_file = os.path.join(calibration_dir, "images.txt")

        camera_params: Dict[int, Tuple[float, float, float, float]] = {}
        if os.path.isfile(cameras_file):
            with open(cameras_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    tokens = line.split()
                    if len(tokens) < 8:
                        continue
                    try:
                        camera_id = int(tokens[0])
                        fx = float(tokens[4])
                        fy = float(tokens[5])
                        cx = float(tokens[6])
                        cy = float(tokens[7])
                    except ValueError:
                        continue
                    camera_params[camera_id] = (fx, fy, cx, cy)

        intrinsics_map: Dict[str, torch.Tensor] = {}
        if os.path.isfile(images_file) and camera_params:
            with open(images_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    tokens = line.split()
                    if len(tokens) < 10:
                        continue
                    try:
                        camera_id = int(tokens[8])
                    except ValueError:
                        continue
                    name = tokens[9]
                    if camera_id not in camera_params:
                        continue
                    fx, fy, cx, cy = camera_params[camera_id]
                    K = torch.tensor(
                        [
                            [fx, 0.0, cx],
                            [0.0, fy, cy],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=torch.float32,
                    )
                    image_abs = os.path.normpath(os.path.join(scene_dir, "images", name))
                    intrinsics_map[image_abs] = K
        return intrinsics_map


class ETH3D_F(SequenceDataset):
    min_depth = 0.05
    max_depth = 60.0
    depth_scale = 1000.0
    test_split = "train.txt"
    train_split = "train.txt"
    sequences_file = "sequences.json"
    hdf5_paths = ["ETH3D-F.hdf5"]

    def __init__(
        self,
        image_shape: tuple[int, int],
        split_file: str,
        test_mode: bool,
        normalize: bool,
        augmentations_db: dict[str, float],
        resize_method: str,
        mini: float = 1.0,
        num_frames: int = 1,
        benchmark: bool = False,
        decode_fields: list[str] = ["image", "depth"],
        inplace_fields: list[str] = ["camera_params", "cam2w"],
        **kwargs,
    ) -> None:
        super().__init__(
            image_shape=image_shape,
            split_file=split_file,
            test_mode=test_mode,
            benchmark=benchmark,
            normalize=normalize,
            augmentations_db=augmentations_db,
            resize_method=resize_method,
            mini=mini,
            num_frames=num_frames,
            decode_fields=(
                decode_fields if not test_mode else [*decode_fields, "points"]
            ),
            inplace_fields=inplace_fields,
            **kwargs,
        )

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [True] * self.num_frames * self.num_copies
        results["quality"] = [1] * self.num_frames * self.num_copies
        return results
