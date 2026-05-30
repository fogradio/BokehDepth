import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torchvision.io import read_image
import torchvision.transforms.v2.functional as TF
from torchvision.transforms import InterpolationMode

from unidepth.datasets.image_dataset import ImageDataset
from unidepth.datasets.utils import DatasetFromList


DEFAULT_FOV = math.radians(60.0)


class HyperSim(ImageDataset):
    min_depth = 0.01
    max_depth = 50.0
    depth_scale = 1000.0
    test_split = "val.txt"
    train_split = "train.txt"
    intrisics_file = "intrinsics.json"
    hdf5_paths = [f"hypersim/hypersim_{i}.hdf5" for i in range(8)]

    def __init__(
        self,
        image_shape,
        split_file,
        test_mode,
        benchmark=False,
        augmentations_db={},
        normalize=True,
        resize_method="hard",
        mini=1.0,
        **kwargs,
    ):
        self.defocus_stack_indices = kwargs.pop("defocus_stack_indices", None)
        if self.defocus_stack_indices is not None:
            self.defocus_stack_indices = [int(i) for i in self.defocus_stack_indices]

        manifest_paths_arg = kwargs.pop("manifest_paths", None)
        if manifest_paths_arg is not None and not isinstance(
            manifest_paths_arg, (list, tuple)
        ):
            manifest_paths_arg = [manifest_paths_arg]
        if manifest_paths_arg is not None:
            manifest_paths = [os.fspath(p) for p in manifest_paths_arg]
        else:
            env_manifests = os.environ.get("HYPERSIM_MANIFEST_PATHS")
            manifest_paths = (
                [p for p in (env_manifests or "").split(os.pathsep) if p]
                if env_manifests
                else None
            )
        self.manifest_path = kwargs.pop("manifest_path", None) or os.environ.get(
            "HYPERSIM_MANIFEST_PATH"
        )
        if manifest_paths is None and self.manifest_path:
            manifest_paths = [self.manifest_path]
        elif manifest_paths and self.manifest_path:
            if self.manifest_path not in manifest_paths:
                manifest_paths.insert(0, self.manifest_path)
        self.manifest_paths: Optional[List[str]] = manifest_paths
        manifest_split = kwargs.pop("manifest_split", None)
        self.camera_metadata_path = kwargs.pop(
            "camera_metadata_path", None
        ) or os.environ.get("HYPERSIM_CAMERA_METADATA")

        # Initialize these attributes before calling super().__init__()
        # because the parent's __init__ calls get_mapper(), which needs _use_manifest.
        self.test_mode = test_mode
        self.manifest_split = (
            manifest_split if manifest_split is not None else ("test" if test_mode else None)
        )
        self._use_manifest = False
        self._camera_metadata: Optional[Dict[str, Dict[str, float]]] = None
        self._camera_loaded = False
        self.manifest_group_indices: List[List[int]] = []
        self.manifest_source_paths: List[str] = []

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

        self.load_dataset()

    def load_dataset(self):
        if self.manifest_paths:
            manifest_paths = [
                self._resolve_path(path) for path in self.manifest_paths
            ]
            missing_paths = [path for path in manifest_paths if not os.path.isfile(path)]
            if missing_paths:
                raise FileNotFoundError(
                    f"HyperSim manifest file(s) not found: {missing_paths}"
                )
            if len(manifest_paths) == 1:
                self._load_manifest_dataset(manifest_paths[0])
            else:
                self._load_multi_manifest_dataset(manifest_paths)
            return

        if self.manifest_path:
            manifest_path = self._resolve_path(self.manifest_path)
            if os.path.isfile(manifest_path):
                self._load_manifest_dataset(manifest_path)
                return
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        h5file = h5py.File(
            os.path.join(self.data_root, self.hdf5_paths[0]),
            "r",
            libver="latest",
            swmr=True,
        )
        txt_file = np.array(h5file[self.split_file])
        txt_string = txt_file.tostring().decode("ascii").strip("\n")
        intrinsics = np.array(h5file[self.intrisics_file]).tostring().decode("ascii")
        intrinsics = json.loads(intrinsics)

        # with open(os.path.join(os.environ["TMPDIR"], self.split_file), "w") as f:
        #     f.write(txt_string)
        # with open(os.path.join(os.environ["TMPDIR"], self.intrisics_file), "w") as f:
        #     json.dump(intrinsics, f)

        dataset = []
        for line in txt_string.split("\n"):
            image_filename, depth_filename, chunk_idx = line.strip().split(" ")
            intrinsics_val = torch.tensor(
                intrinsics[os.path.join(*image_filename.split("/")[:2])]
            ).squeeze()[:, :3]
            sample = [image_filename, depth_filename, intrinsics_val, chunk_idx]
            dataset.append(sample)
        h5file.close()

        if not self.test_mode:
            dataset = self.chunk(dataset, chunk_dim=1, pct=self.mini)

        if self.test_mode and not self.benchmark:  # corresponds to 712 images
            dataset = self.chunk(dataset, chunk_dim=1, pct=0.1)

        self.dataset = DatasetFromList(dataset)
        self.log_load_dataset()

    def get_mapper(self):
        if self._use_manifest:
            return {}
        return {
            "image_filename": 0,
            "depth_filename": 1,
            "K": 2,
            "chunk_idx": 3,
        }

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [True] * self.num_copies
        results["synthetic"] = [True] * self.num_copies
        results["quality"] = [0] * self.num_copies
        return results

    def get_single_item(self, idx, sample=None, mapper=None):
        if self._use_manifest:
            return self._get_single_item_from_manifest(idx)
        return super().get_single_item(idx, sample=sample, mapper=mapper)

    def _get_single_item_from_manifest(self, idx: int):
        sample: Dict[str, Any] = self.dataset[idx]
        image_path: str = sample["image_path"]
        depth_path: str = sample["depth_path"]

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

        image_tensor = self._load_hdr_image(image_path)
        seq["filename"] = image_path
        seq["image_fields"].add("image")
        seq["image_ori_shape"] = image_tensor.shape[-2:]
        seq["image"] = image_tensor.unsqueeze(0)

        stack_paths: List[str] = sample.get("stack_paths", [])
        if self.defocus_stack_indices and stack_paths:
            selected_paths, selected_k = self._select_stack_indices(
                stack_paths, sample.get("k_values", [])
            )
            if selected_paths:
                stack_paths = selected_paths
                sample["k_values"] = selected_k if selected_k else sample.get("k_values", [])

        if stack_paths:
            stack_tensor, k_tensor = self._load_stack_frames(
                stack_paths,
                image_tensor.shape[-2:],
                sample.get("k_values"),
            )
            seq["defocus_stack"] = stack_tensor
            seq["image_fields"].add("defocus_stack")
            if k_tensor is not None:
                seq["k_values"] = k_tensor

        depth_array = self._load_depth(depth_path)
        finite_mask = np.isfinite(depth_array)
        positive_mask = depth_array > self.min_depth
        valid_mask = finite_mask & positive_mask
        if not valid_mask.any():
            raise ValueError(f"HyperSim depth map {depth_path} has no valid positive values.")

        seq["gt_fields"].add("depth")
        depth_tensor = torch.from_numpy(np.where(valid_mask, depth_array, 0.0))
        seq["depth_ori_shape"] = depth_tensor.shape
        seq["depth"] = depth_tensor.view(1, 1, *depth_tensor.shape)

        validity = torch.from_numpy(valid_mask.astype(np.float32))
        mask_fields = seq.setdefault("mask_fields", set())
        mask_fields.update({"validity_mask", "depth_mask"})
        reshaped_mask = validity.view(1, 1, *depth_tensor.shape)
        seq["validity_mask"] = reshaped_mask
        seq["depth_mask"] = reshaped_mask.clone()

        seq["metadata"] = sample.get("metadata", {}).copy()
        if sample.get("stack_index"):
            seq["metadata"]["stack_index_path"] = sample["stack_index"]
        if sample.get("source_depth"):
            seq["metadata"]["source_depth_path"] = sample["source_depth"]

        seq["camera_fields"].update({"camera", "cam2w"})
        K = self._get_intrinsics_from_sample(sample, image_tensor.shape[-2:])
        camera = torch.zeros((1, 3, 3), dtype=K.dtype)
        camera[0] = K
        seq["camera"] = self._build_camera(camera)
        seq["cam2w"] = torch.eye(4, dtype=K.dtype).unsqueeze(0)

        results = self.preprocess(results)
        if not self.test_mode:
            results = self.augment(results)
        results = self.postprocess(results)
        return results

    def _build_camera(self, K: torch.Tensor):
        from unidepth.utils.camera import BatchCamera, Pinhole

        return BatchCamera.from_camera(Pinhole(K=K.clone()))

    def _resolve_path(self, path: str) -> str:
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return path
        return os.path.join(self.data_root, path)

    def _select_stack_indices(
        self,
        stack_paths: List[str],
        k_values: Iterable[float],
    ) -> Tuple[List[str], List[float]]:
        k_values = list(k_values) if k_values is not None else []
        selected_paths: List[str] = []
        selected_k: List[float] = []
        for idx in self.defocus_stack_indices:
            if 0 <= idx < len(stack_paths):
                selected_paths.append(stack_paths[idx])
                if k_values and idx < len(k_values):
                    selected_k.append(k_values[idx])
        return selected_paths, selected_k

    def _load_manifest_dataset(self, manifest_path: str) -> None:
        records = self._parse_manifest_records(manifest_path)
        if not records:
            raise FileNotFoundError(
                f"Manifest parsed from {manifest_path} but no valid HyperSim samples were found."
            )

        if not self.test_mode:
            records = self.chunk(records, chunk_dim=1, pct=self.mini)

        self.dataset = DatasetFromList(records, serialize=False)
        self._use_manifest = True
        self.depth_scale = 1.0
        self.manifest_group_indices = []
        self.manifest_source_paths = [manifest_path]
        self.log_load_dataset()

    def _load_multi_manifest_dataset(self, manifest_paths: Sequence[str]) -> None:
        manifest_records: List[List[Dict[str, Any]]] = []
        for path in manifest_paths:
            records = self._parse_manifest_records(path)
            if not records:
                raise FileNotFoundError(
                    f"Manifest parsed from {path} but no valid HyperSim samples were found."
                )
            if not self.test_mode:
                records = self.chunk(records, chunk_dim=1, pct=self.mini)
            manifest_records.append(records)

        combined: List[Dict[str, Any]] = []
        group_indices: List[List[int]] = []
        for records in manifest_records:
            start = len(combined)
            combined.extend(records)
            group_indices.append(list(range(start, start + len(records))))

        self.dataset = DatasetFromList(combined, serialize=False)
        self._use_manifest = True
        self.depth_scale = 1.0
        # Only expose manifest groups when we actually have multiple manifests
        self.manifest_group_indices = [g for g in group_indices if g]
        self.manifest_source_paths = list(manifest_paths)
        self.log_load_dataset()

    def _parse_manifest_records(self, manifest_path: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        desired_split = (
            self.manifest_split.lower() if self.manifest_split else None
        )

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry: Dict[str, Any] = json.loads(line)
                except json.JSONDecodeError:
                    continue

                split_name = entry.get("split")
                if desired_split and split_name and split_name.lower() != desired_split:
                    continue

                image_path = entry.get("ref") or entry.get("source_rgb")
                depth_path = entry.get("depth") or entry.get("source_depth")
                if not image_path or not depth_path:
                    continue

                image_path = self._resolve_path(image_path)
                depth_path = self._resolve_path(depth_path)
                if not os.path.isfile(image_path) or not os.path.isfile(depth_path):
                    continue

                stack_paths = [
                    self._resolve_path(p) for p in entry.get("stack", []) or []
                ]
                stack_paths = [p for p in stack_paths if os.path.isfile(p)]

                k_values = entry.get("k") or []
                if isinstance(k_values, dict):
                    k_values = [k_values[k] for k in sorted(k_values.keys())]

                stack_index_path = entry.get("stack_index")
                if stack_index_path:
                    stack_index_path = self._resolve_path(stack_index_path)
                    if not os.path.isfile(stack_index_path):
                        stack_index_path = None

                source_depth = entry.get("source_depth")
                if source_depth:
                    source_depth = self._resolve_path(source_depth)
                    if not os.path.isfile(source_depth):
                        source_depth = None

                scene_name = entry.get("scene")
                camera_name = entry.get("camera", "cam_00")
                source_size = entry.get("source_size")
                if isinstance(source_size, list) and len(source_size) == 2:
                    width, height = source_size
                else:
                    width, height = 1024, 768

                record = {
                    "image_path": image_path,
                    "depth_path": depth_path,
                    "stack_paths": stack_paths,
                    "k_values": k_values,
                    "stack_index": stack_index_path,
                    "source_depth": source_depth,
                    "scene": scene_name,
                    "camera": camera_name,
                    "source_size": (int(height), int(width)),
                    "metadata": entry,
                }

                records.append(record)

        for record in records:
            record["manifest_source"] = manifest_path
            if isinstance(record.get("metadata"), dict):
                record["metadata"].setdefault("manifest_path", manifest_path)

        return records

    def _tone_map_hdr(self, hdr: np.ndarray) -> np.ndarray:
        """HDR-to-LDR tone mapping, robust against extreme values."""
        hdr = np.asarray(hdr, dtype=np.float32)

        # Critical fix: clean NaN/Inf in the HDR data.
        if not np.all(np.isfinite(hdr)):
            nan_count = (~np.isfinite(hdr)).sum()
            # Replace NaN with 0 and Inf with a finite large value.
            hdr = np.nan_to_num(hdr, nan=0.0, posinf=10.0, neginf=0.0)
            if hasattr(self, '_tone_map_warning_count'):
                self._tone_map_warning_count += 1
            else:
                self._tone_map_warning_count = 1

        # Clip extreme values to avoid downstream issues.
        hdr = np.clip(hdr, 0.0, 100.0)  # HDR values typically never exceed 100

        # Reinhard tone mapping.
        ldr = hdr / (hdr + 1.0)
        ldr = np.clip(ldr, 0.0, 1.0)

        # Gamma correction.
        ldr = np.power(ldr, 1.0 / 2.2)

        # Convert to uint8.
        ldr = (ldr * 255.0).round().astype(np.uint8)
        return ldr

    def _load_hdr_image(self, path: str) -> torch.Tensor:
        """Load an HDR image and convert it to an LDR tensor, with robust error handling."""
        try:
            with h5py.File(path, "r") as f:
                data = np.array(f["dataset"], dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"failed to load HDR data from {path}: {e}")

        # Check the data shape.
        if data.ndim != 3 or data.shape[2] != 3:
            raise ValueError(f"HDR image {path} has unexpected shape: {data.shape}, expected (H, W, 3)")

        ldr = self._tone_map_hdr(data)
        tensor = torch.from_numpy(ldr).permute(2, 0, 1).contiguous()

        # Final validation: make sure the tensor is valid.
        if not torch.all(torch.isfinite(tensor.float())):
            raise ValueError(f"image {path} still contains non-finite values after tone mapping")

        return tensor

    def _load_stack_frames(
        self,
        stack_paths: List[str],
        target_hw: Tuple[int, int],
        k_values: Optional[Iterable[float]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Load defocus-stack frames with robust error handling."""
        frames: List[torch.Tensor] = []
        for i, stack_path in enumerate(stack_paths):
            try:
                frame = read_image(stack_path)
            except Exception as e:
                raise RuntimeError(f"failed to load stack frame {stack_path}: {e}")

            if frame.shape[0] == 1:
                frame = frame.repeat(3, 1, 1)

            # Check for invalid values (uint8 reads can still fail).
            if not torch.all(torch.isfinite(frame.float())):
                raise ValueError(f"stack frame {stack_path} contains non-finite values")

            if frame.shape[-2:] != target_hw:
                frame = TF.resize(
                    frame.float().unsqueeze(0) / 255.0,
                    target_hw,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ).squeeze(0)
                frame = torch.clamp(frame * 255.0, 0.0, 255.0).to(torch.uint8)
            frames.append(frame)

        stack_tensor = torch.stack(frames, dim=0)
        k_tensor = None
        if k_values:
            k_list = list(k_values)
            # Validate the k values.
            if not all(isinstance(k, (int, float)) and math.isfinite(k) for k in k_list):
                raise ValueError(f"k_values contains invalid entries: {k_list}")
            k_tensor = torch.tensor(k_list, dtype=torch.float32)
        return stack_tensor, k_tensor

    def _load_depth(self, path: str) -> np.ndarray:
        """Load the depth map with extra robustness checks."""
        try:
            with h5py.File(path, "r") as f:
                depth = np.array(f["dataset"], dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"failed to load depth data from {path}: {e}")

        # Shape check.
        if depth.ndim != 2:
            raise ValueError(f"depth map {path} has unexpected shape: {depth.shape}, expected (H, W)")

        # Clean extreme outliers (keep depth within a reasonable range).
        # HyperSim depth typically lives in [0.01, 50.0] meters.
        if np.any(depth > 1e6):
            # Treat absurdly large values as invalid.
            depth = np.where(depth > 1e6, np.nan, depth)

        return depth

    def _load_camera_metadata(self) -> None:
        if self._camera_loaded:
            return
        metadata_path = None
        if self.camera_metadata_path:
            metadata_path = self._resolve_path(self.camera_metadata_path)
        else:
            candidate = os.path.join(
                self.data_root,
                "ml-hypersim",
                "contrib",
                "mikeroberts3000",
                "metadata_camera_parameters.csv",
            )
            if os.path.isfile(candidate):
                metadata_path = candidate
            else:
                metadata_path = self._derive_metadata_path_from_manifest()

        camera_dict: Dict[str, Dict[str, float]] = {}
        if metadata_path and os.path.isfile(metadata_path):
            with open(metadata_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    scene_name = row.get("scene_name")
                    if not scene_name:
                        continue
                    try:
                        width = float(row.get("settings_output_img_width") or 1024.0)
                        height = float(row.get("settings_output_img_height") or 768.0)
                        fov = float(row.get("settings_camera_fov") or DEFAULT_FOV)
                    except ValueError:
                        width, height, fov = 1024.0, 768.0, DEFAULT_FOV
                    camera_dict[scene_name] = {
                        "width": width,
                        "height": height,
                        "fov": fov,
                    }
        self._camera_metadata = camera_dict if camera_dict else {}
        self._camera_loaded = True

    def _derive_metadata_path_from_manifest(self) -> Optional[str]:
        if not self.manifest_path:
            return None
        manifest_abs = self._resolve_path(self.manifest_path)
        current = os.path.dirname(manifest_abs)
        while current and current != os.path.dirname(current):
            if os.path.basename(current) == "ml-hypersim":
                candidate = os.path.join(
                    current,
                    "contrib",
                    "mikeroberts3000",
                    "metadata_camera_parameters.csv",
                )
                if os.path.isfile(candidate):
                    return candidate
            current = os.path.dirname(current)
        return None

    def _get_intrinsics_from_sample(
        self, sample: Dict[str, Any], image_hw: Tuple[int, int]
    ) -> torch.Tensor:
        self._load_camera_metadata()
        scene = sample.get("scene")
        default_size = (float(image_hw[0]), float(image_hw[1]))
        source_size = sample.get("source_size", default_size)
        height = float(source_size[0])
        width = float(source_size[1])
        metadata = self._camera_metadata or {}
        camera_entry = metadata.get(scene) if scene else None
        fov = camera_entry["fov"] if camera_entry else DEFAULT_FOV
        width = camera_entry["width"] if camera_entry else width
        height = camera_entry["height"] if camera_entry else height

        fy = 0.5 * height / math.tan(0.5 * fov)
        fx = fy * (width / height)
        cx = width / 2.0
        cy = height / 2.0

        K = torch.tensor(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        sample["K"] = K
        return K
