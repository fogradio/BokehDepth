import json
import os
from typing import Any, Dict

import h5py
import numpy as np
import torch
from torchvision.io import read_image
from PIL import Image

from unidepth.datasets.image_dataset import ImageDataset
from unidepth.datasets.pipelines import AnnotationMask
from unidepth.datasets.utils import DatasetFromList
from unidepth.utils import identity
from unidepth.utils.camera import BatchCamera, Pinhole


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEBUG_LOG_ENABLED = _env_flag("UNIDEPTH_DEBUG")


class NYUv2Depth(ImageDataset):
    NYU_DISPARITY_SCALE = 10000.0
    CAM_INTRINSIC = {
        "ALL": torch.tensor(
            [
                [5.1885790117450188e02, 0, 3.2558244941119034e02],
                [0, 5.1946961112127485e02, 2.5373616633400465e02],
                [0, 0, 1],
            ]
        )
    }
    min_depth = 0.005
    max_depth = 10.0
    depth_scale = 1000.0
    log_mean = 0.9140
    log_std = 0.4825
    test_split = "nyu_test.txt"
    train_split = "nyu_train.txt"
    hdf5_paths = ["nyuv2.hdf5"]

    def __init__(
        self,
        image_shape,
        split_file,
        test_mode,
        crop=None,
        benchmark=False,
        augmentations_db={},
        normalize=True,
        resize_method="hard",
        mini=1.0,
        manifest_path: str | None = None,
        manifest_depth_mode: str = "auto",
        official_root: str | None = None,
        **kwargs,
    ):
        self.defocus_stack_indices = kwargs.pop("defocus_stack_indices", None)
        if self.defocus_stack_indices is not None:
            self.defocus_stack_indices = [int(i) for i in self.defocus_stack_indices]

        self.debug_augmentation = kwargs.pop("debug_augmentation", False)
        self.manifest_path = manifest_path or os.environ.get("NYUV2_MANIFEST_PATH")
        self._use_manifest = False
        self.manifest_depth_mode = manifest_depth_mode
        self.official_root = official_root

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
        self.masker = AnnotationMask(
            min_value=0.0,
            max_value=self.max_depth if test_mode else None,
            custom_fn=self.eval_mask if test_mode else lambda x, *args, **kwargs: x,
        )
        self.test_mode = test_mode
        self.load_dataset()

    def _debug(self, *args, **kwargs):
        if self.debug_augmentation or DEBUG_LOG_ENABLED:
            print(*args, **kwargs)

    def load_dataset(self):
        manifest_path = self.manifest_path
        if manifest_path:
            manifest_path = os.path.expanduser(manifest_path)
            if not os.path.isabs(manifest_path):
                manifest_path = os.path.join(self.data_root, manifest_path)
            if os.path.isfile(manifest_path):
                self._load_manifest_dataset(manifest_path)
                return
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        if self.official_root:
            official_root = os.path.expanduser(self.official_root)
            if not os.path.isabs(official_root):
                official_root = os.path.join(self.data_root, official_root)
            if os.path.isdir(official_root):
                self._load_official_png_dataset(official_root)
                return
            raise FileNotFoundError(f"Official NYUv2 directory not found: {official_root}")

        h5file = h5py.File(
            os.path.join(self.data_root, self.hdf5_paths[0]),
            "r",
            libver="latest",
            swmr=True,
        )
        txt_file = np.array(h5file[self.split_file])
        txt_string = txt_file.tostring().decode("ascii")[:-1]  # correct the -1
        h5file.close()
        dataset = []
        for line in txt_string.split("\n"):
            image_filename, depth_filename, _ = line.strip().split(" ")
            sample = [
                image_filename,
                depth_filename,
            ]
            dataset.append(sample)

        if not self.test_mode:
            dataset = self.chunk(dataset, chunk_dim=1, pct=self.mini)

        self.dataset = DatasetFromList(dataset)
        self.log_load_dataset()

    def _load_manifest_dataset(self, manifest_path: str) -> None:
        records: list[Dict[str, Any]] = []
        desired_split = "test" if self.test_mode else "train"

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
                if split_name and split_name.lower() != desired_split:
                    continue

                image_path = entry.get("ref")
                depth_path = entry.get("depth")
                if not image_path or not depth_path:
                    continue

                image_path = os.path.expanduser(image_path)
                depth_path = os.path.expanduser(depth_path)
                if not os.path.isabs(image_path):
                    image_path = os.path.join(self.data_root, image_path)
                if not os.path.isabs(depth_path):
                    depth_path = os.path.join(self.data_root, depth_path)

                if not os.path.isfile(image_path) or not os.path.isfile(depth_path):
                    continue

                stack_paths = entry.get("stack", [])
                k_values = entry.get("k", [])
                stack_index_path = entry.get("stack_index")
                source_depth_path = entry.get("source_depth")

                valid_stack_paths: list[str] = []
                for stack_path in stack_paths:
                    stack_abs = os.path.expanduser(stack_path)
                    if not os.path.isabs(stack_abs):
                        stack_abs = os.path.join(self.data_root, stack_abs)
                    if os.path.isfile(stack_abs):
                        valid_stack_paths.append(stack_abs)

                if self.defocus_stack_indices and valid_stack_paths:
                    selected_paths: list[str] = []
                    selected_k: list[float] = []
                    for idx in self.defocus_stack_indices:
                        if 0 <= idx < len(valid_stack_paths):
                            selected_paths.append(valid_stack_paths[idx])
                            if k_values and idx < len(k_values):
                                selected_k.append(k_values[idx])
                    if selected_paths:
                        valid_stack_paths = selected_paths
                        if k_values:
                            k_values = selected_k if selected_k else []

                records.append(
                    {
                        "image_path": image_path,
                        "depth_path": depth_path,
                        "stack_paths": valid_stack_paths,
                        "k_values": k_values,
                        "stack_index": stack_index_path,
                        "source_depth": source_depth_path,
                        "metadata": entry,
                    }
                )

        if not records:
            raise FileNotFoundError(
                f"Manifest parsed from {manifest_path} but no valid NYUv2 samples were found."
            )

        if not self.test_mode:
            records = self.chunk(records, chunk_dim=1, pct=self.mini)

        self.dataset = DatasetFromList(records, serialize=False)
        self._use_manifest = True
        self.depth_scale = 1.0
        self.manifest_path = manifest_path
        self.log_load_dataset()

    def _load_official_png_dataset(self, official_root: str) -> None:
        records: list[Dict[str, Any]] = []
        rgb_files = sorted(
            fname
            for fname in os.listdir(official_root)
            if fname.lower().startswith("rgb_") and fname.lower().endswith(".png")
        )

        for rgb_file in rgb_files:
            stem = rgb_file.split("rgb_")[-1].split(".")[0]
            depth_file = f"depth_{stem}.png"
            rgb_path = os.path.join(official_root, rgb_file)
            depth_path = os.path.join(official_root, depth_file)
            if not os.path.isfile(depth_path):
                continue

            records.append(
                {
                    "image_path": rgb_path,
                    "depth_path": depth_path,
                    "stack_paths": [],
                    "k_values": [],
                    "stack_index": None,
                    "source_depth": depth_path,
                    "metadata": {"split": "official", "frame": stem},
                }
            )

        if not records:
            raise FileNotFoundError(
                f"No matching rgb_*.png/depth_*.png pairs found in {official_root}"
            )

        if not self.test_mode:
            records = self.chunk(records, chunk_dim=1, pct=self.mini)

        self.dataset = DatasetFromList(records, serialize=False)
        self._use_manifest = True
        self.depth_scale = 1.0
        self.manifest_path = None
        self.log_load_dataset()

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [True] * self.num_copies
        return results

    def get_intrinsics(self, idx, image_name):
        return self.CAM_INTRINSIC["ALL"].clone()

    def eval_mask(self, valid_mask, info={}):
        border_mask = torch.zeros_like(valid_mask)
        border_mask[..., 45:-9, 41:-39] = 1
        return torch.logical_and(valid_mask, border_mask)

    def get_mapper(self):
        return {
            "image_filename": 0,
            "depth_filename": 1,
        }

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [True] * self.num_copies
        results["quality"] = [2] * self.num_copies
        return results

    def get_single_item(self, idx, sample=None, mapper=None):
        if getattr(self, "_use_manifest", False):
            return self._get_single_item_from_manifest(idx)
        return super().get_single_item(idx, sample=sample, mapper=mapper)

    def _get_single_item_from_manifest(self, idx: int):
        sample = self.dataset[idx]
        image_path = sample["image_path"]
        depth_path = sample["depth_path"]

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

        image_tensor = read_image(image_path)
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)
        seq["filename"] = image_path
        seq["image_fields"].add("image")
        seq["image_ori_shape"] = image_tensor.shape[-2:]
        seq["image"] = image_tensor.unsqueeze(0)

        stack_paths = sample.get("stack_paths", [])
        k_values = sample.get("k_values", [])
        if stack_paths:
            stack_tensors = []
            for stack_path in stack_paths:
                stack_img = read_image(stack_path)
                if stack_img.shape[0] == 1:
                    stack_img = stack_img.repeat(3, 1, 1)
                stack_tensors.append(stack_img)
            seq["defocus_stack"] = torch.stack(stack_tensors, dim=0)
            seq["image_fields"].add("defocus_stack")
            seq["k_values"] = (
                torch.tensor(k_values, dtype=torch.float32)
                if k_values
                else None
            )

        depth_array = self._load_depth_file(depth_path)
        depth_array = self._decode_manifest_depth(depth_array)

        finite_mask = np.isfinite(depth_array)
        if not finite_mask.any():
            raise ValueError(f"NYUv2 depth map {depth_path} has no finite depth values")
        depth_array = np.where(finite_mask, depth_array, 0.0)

        positive_mask = depth_array > self.min_depth
        if not positive_mask.any():
            raise ValueError(
                f"NYUv2 depth map {depth_path} has no positive values after decoding"
            )

        depth_tensor = torch.from_numpy(depth_array)
        seq["gt_fields"].add("depth")
        seq["depth_ori_shape"] = depth_tensor.shape
        seq["depth"] = depth_tensor.view(1, 1, *depth_tensor.shape)

        validity_mask = torch.from_numpy((finite_mask & positive_mask).astype(np.float32))
        mask_fields = seq.setdefault("mask_fields", set())
        mask_fields.update({"validity_mask", "depth_mask"})
        reshaped_mask = validity_mask.view(1, 1, *depth_tensor.shape)
        seq["validity_mask"] = reshaped_mask
        seq["depth_mask"] = reshaped_mask.clone()

        seq["metadata"] = sample.get("metadata", {}).copy()
        if sample.get("stack_index"):
            seq["metadata"]["stack_index_path"] = sample["stack_index"]
        if sample.get("source_depth"):
            seq["metadata"]["source_depth_path"] = sample["source_depth"]

        seq["camera_fields"].update({"camera", "cam2w"})
        K = self.get_intrinsics(idx, image_path)
        if K is None:
            K = self.CAM_INTRINSIC["ALL"].clone()
        camera = BatchCamera.from_camera(Pinhole(K=K.unsqueeze(0)))
        seq["camera"] = camera
        seq["cam2w"] = torch.eye(4, dtype=K.dtype).unsqueeze(0)

        if (self.debug_augmentation or DEBUG_LOG_ENABLED) and "defocus_stack" in seq:
            self._debug("=== NYUv2 Before Preprocess ===")
            self._debug(f"image shape: {seq['image'].shape}")
            self._debug(f"defocus stack shape: {seq['defocus_stack'].shape}")

        results = self.preprocess(results)

        if (self.debug_augmentation or DEBUG_LOG_ENABLED) and "defocus_stack" in results:
            self._debug("=== NYUv2 After Preprocess ===")
            for seq_key in results.get("sequence_fields", []):
                seq_item = results.get(seq_key, {})
                if isinstance(seq_item, dict):
                    self._debug(f"image shape: {seq_item.get('image', torch.empty(0)).shape}")
                    if "defocus_stack" in seq_item:
                        self._debug(f"defocus stack shape: {seq_item['defocus_stack'].shape}")

        if not self.test_mode:
            results = self.augment(results)

        results = self.postprocess(results)

        if (self.debug_augmentation or DEBUG_LOG_ENABLED) and "seq0" in results:
            self._debug("=== NYUv2 Manifest Postprocess Summary ===")
            seq0 = results.get("seq0", {})
            self._debug(f"  image shape: {seq0.get('image', torch.empty(0)).shape}")
            if "defocus_stack" in seq0:
                self._debug(f"  defocus stack shape: {seq0['defocus_stack'].shape}")

        return results

    def _load_depth_file(self, depth_path: str) -> np.ndarray:
        ext = os.path.splitext(depth_path)[1].lower()
        if ext == ".npy":
            depth_array = np.load(depth_path)
        elif ext == ".npz":
            data = np.load(depth_path)
            if "depth" in data:
                depth_array = data["depth"]
            else:
                first_key = list(data.keys())[0]
                depth_array = data[first_key]
        elif ext in {".png", ".jpg", ".jpeg"}:
            depth_array = np.array(Image.open(depth_path), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported depth file extension for NYUv2: {depth_path}")
        return np.asarray(depth_array, dtype=np.float32)

    def _decode_manifest_depth(self, depth: np.ndarray) -> np.ndarray:
        """Decode manifest-provided NYUv2 depth into metres.

        The bokeh-diffusion preprocessing saved the official ``depth_XXXXX.png`` disparity
        encoding (values roughly 6k-60k). To stay consistent with community evaluation
        protocols we only support converting disparity to meters via
        ``depth_m = 10000 / disparity``. If the raw array is already within the
        0~max_depth range it is returned as meters directly.
        """

        if depth.size == 0:
            return depth

        finite = np.isfinite(depth)
        if not finite.any():
            return depth

        safe = finite & (depth > 0)
        if not safe.any():
            return depth.astype(np.float32, copy=False)

        raw = depth[safe].astype(np.float32, copy=False)
        mode = (self.manifest_depth_mode or "auto").lower()

        if mode in {"raw", "meters", "metres"}:
            decoded = raw
        else:
            decoded = np.zeros_like(raw, dtype=np.float32)
            nonzero = raw > 0
            decoded[nonzero] = self.NYU_DISPARITY_SCALE / raw[nonzero]

        depth_out = np.zeros_like(depth, dtype=np.float32)
        depth_out[safe] = decoded

        return np.clip(depth_out, a_min=0.0, a_max=None)
