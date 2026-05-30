import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torchvision.io import read_image

from unidepth.datasets.image_dataset import ImageDataset
from unidepth.datasets.pipelines import AnnotationMask
from unidepth.datasets.sequence_dataset import SequenceDataset
from unidepth.datasets.utils import DatasetFromList
from unidepth.utils.camera import BatchCamera, Pinhole


class VKITTI(SequenceDataset):
    min_depth = 0.01
    max_depth = 80.0  # Match KITTI; prevents extreme depth values from crashing training
    depth_scale = 256.0
    test_split = "training.txt"
    train_split = "training.txt"
    sequences_file = "sequences.json"
    hdf5_paths = ["VKITTI2.hdf5"]

    def __init__(
        self,
        image_shape: tuple[int, int],
        split_file: str,
        test_mode: bool,
        normalize: bool,
        augmentations_db: dict[str, Any],
        resize_method: str,
        mini: float = 1.0,
        num_frames: int = 1,
        benchmark: bool = False,
        decode_fields: list[str] = ["image", "depth", "flow_fwd", "flow_fwd_mask"],
        inplace_fields: list[str] = ["K", "cam2w"],
        manifest_path: str | None = None,
        **kwargs,
    ) -> None:
        self.defocus_stack_indices = kwargs.pop("defocus_stack_indices", None)
        if self.defocus_stack_indices is not None:
            self.defocus_stack_indices = [int(i) for i in self.defocus_stack_indices]

        self.manifest_path = manifest_path or os.environ.get("VKITTI_MANIFEST_PATH")
        self._use_manifest = False

        # If a manifest path is supplied, initialize through BaseDataset directly (bypassing ImageDataset/SequenceDataset).
        if self.manifest_path:
            # Call BaseDataset's init directly.
            from unidepth.datasets.base_dataset import BaseDataset
            BaseDataset.__init__(
                self,
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
            # Reset the masker with VKITTI-specific depth bounds.
            # max_value is also needed during training to filter extreme depths.
            self.masker = AnnotationMask(
                min_value=self.min_depth,
                max_value=self.max_depth,  # use max_depth for both training and test
            )
            self.load_dataset()
        else:
            # Use the original HDF5 sequence-dataset path.
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
                decode_fields=decode_fields,
                inplace_fields=inplace_fields,
                **kwargs,
            )

    def load_dataset(self):
        """Pick the load path depending on whether a manifest is supplied."""
        if self.manifest_path:
            manifest_path = os.path.expanduser(self.manifest_path)
            if not os.path.isabs(manifest_path):
                manifest_path = os.path.join(self.data_root, manifest_path)

            if os.path.isfile(manifest_path):
                self._load_manifest_dataset(manifest_path)
                return
            else:
                raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        else:
            # Fall back to the parent class HDF5 loader.
            super().load_dataset()

    def _load_manifest_dataset(self, manifest_path: str) -> None:
        """Load VKITTI2 defocus-stack data from a JSONL manifest."""
        records = []
        self.invalid_depth_num = 0

        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry: Dict[str, Any] = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # VKITTI2 jsonl fields: ref, stack, k, depth, mask
                image_path = entry.get("ref")
                # Prefer source_depth (original depth path); fall back to depth if missing.
                depth_path = entry.get("source_depth") or entry.get("depth")

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

                # Process defocus-stack data.
                stack_paths = entry.get("stack", [])
                k_values = entry.get("k", [])
                mask_path = entry.get("mask", None)

                # Validate the defocus-stack paths.
                valid_stack_paths = []
                if stack_paths:
                    for stack_path in stack_paths:
                        stack_path = os.path.expanduser(stack_path)
                        if not os.path.isabs(stack_path):
                            stack_path = os.path.join(self.data_root, stack_path)
                        if os.path.isfile(stack_path):
                            valid_stack_paths.append(stack_path)

                # Process the mask path.
                if mask_path:
                    mask_path = os.path.expanduser(mask_path)
                    if not os.path.isabs(mask_path):
                        mask_path = os.path.join(self.data_root, mask_path)
                    if not os.path.isfile(mask_path):
                        mask_path = None

                if self.defocus_stack_indices and valid_stack_paths:
                    selected_paths = []
                    selected_k = []
                    for idx in self.defocus_stack_indices:
                        if 0 <= idx < len(valid_stack_paths):
                            if k_values and idx >= len(k_values):
                                continue
                            selected_paths.append(valid_stack_paths[idx])
                            if k_values:
                                selected_k.append(k_values[idx])
                    if selected_paths:
                        valid_stack_paths = selected_paths
                        if k_values:
                            k_values = selected_k if selected_k else []

                records.append(
                    {
                        "image_path": image_path,
                        "depth_path": depth_path,
                        "mask_path": mask_path,
                        "stack_paths": valid_stack_paths,
                        "k_values": k_values,
                        "metadata": entry,
                    }
                )

        if not records:
            raise FileNotFoundError(
                f"Manifest parsed from {manifest_path} but no valid samples were found."
            )

        if not self.test_mode:
            records = self.chunk(records, chunk_dim=1, pct=self.mini)

        self.dataset = DatasetFromList(records, serialize=False)
        self._use_manifest = True
        self.depth_scale = 1.0
        self.manifest_path = manifest_path
        self.log_load_dataset()

    def get_single_item(self, idx, sample=None, mapper=None):
        """Pick the load path depending on whether a manifest is used."""
        if self._use_manifest:
            return self._get_single_item_from_manifest(idx)
        return super().get_single_item(idx, sample=sample, mapper=mapper)

    def _get_single_item_from_manifest(self, idx: int):
        """Load a single sample from the manifest (including the defocus stack)."""
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

        # Load the all-in-focus image.
        image_tensor = read_image(image_path)
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)
        seq["filename"] = image_path
        seq["image_fields"].add("image")
        seq["image_ori_shape"] = image_tensor.shape[-2:]
        seq["image"] = image_tensor.unsqueeze(0)

        # Load defocus-stack images.
        stack_paths = sample.get("stack_paths", [])
        k_values = sample.get("k_values", [])
        if self.defocus_stack_indices and stack_paths:
            selected_paths = []
            selected_k = []
            for idx in self.defocus_stack_indices:
                if 0 <= idx < len(stack_paths):
                    if k_values and idx >= len(k_values):
                        continue
                    selected_paths.append(stack_paths[idx])
                    if k_values:
                        selected_k.append(k_values[idx])
            if selected_paths:
                stack_paths = selected_paths
                if k_values:
                    if selected_k:
                        k_values = selected_k
                    else:
                        k_values = []
        if stack_paths:
            stack_tensors = []
            for stack_path in stack_paths:
                stack_img = read_image(stack_path)
                if stack_img.shape[0] == 1:
                    stack_img = stack_img.repeat(3, 1, 1)
                stack_tensors.append(stack_img)
            # stack shape: [num_stack, C, H, W]
            seq["defocus_stack"] = torch.stack(stack_tensors, dim=0)
            seq["image_fields"].add("defocus_stack")
            # k_values: [num_stack]
            seq["k_values"] = torch.tensor(k_values, dtype=torch.float32) if k_values else None

        # Load the depth map.
        depth_array = self._load_depth_from_path(depth_path)
        depth_tensor = torch.from_numpy(depth_array.astype(np.float32))
        seq["gt_fields"].add("depth")
        seq["depth_ori_shape"] = depth_tensor.shape
        seq["depth"] = depth_tensor.view(1, 1, *depth_tensor.shape)

        # Load the mask if present.
        mask_path = sample.get("mask_path")
        if mask_path:
            mask_tensor = read_image(mask_path)
            if mask_tensor.shape[0] > 1:
                mask_tensor = mask_tensor[0:1]  # keep only the first channel
            seq["mask"] = mask_tensor.unsqueeze(0).float() / 255.0
            seq["mask_fields"].add("mask")

        metadata = sample.get("metadata", {}) or {}
        vkitti_text = metadata.get("vkitti_text") if isinstance(metadata, dict) else None

        K: torch.Tensor | None = None
        cam2w_matrix: torch.Tensor | None = None
        if isinstance(vkitti_text, dict):
            intrinsic = vkitti_text.get("intrinsic")
            if intrinsic:
                fx = float(intrinsic.get("fx", 0.0))
                fy = float(intrinsic.get("fy", 0.0))
                cx = float(intrinsic.get("cx", 0.0))
                cy = float(intrinsic.get("cy", 0.0))
                if fx > 0 and fy > 0:
                    K = torch.tensor(
                        [
                            [fx, 0.0, cx],
                            [0.0, fy, cy],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=torch.float32,
                    )
            extrinsic = vkitti_text.get("extrinsic")
            if extrinsic is not None:
                extrinsic_tensor = torch.tensor(extrinsic, dtype=torch.float32)
                if extrinsic_tensor.ndim == 2 and extrinsic_tensor.shape == (3, 4):
                    extrinsic_full = torch.eye(4, dtype=torch.float32)
                    extrinsic_full[:3, :4] = extrinsic_tensor
                elif extrinsic_tensor.shape == (4, 4):
                    extrinsic_full = extrinsic_tensor
                else:
                    extrinsic_full = None

                if extrinsic_full is not None:
                    try:
                        cam2w_matrix = torch.linalg.inv(extrinsic_full)
                    except RuntimeError:
                        cam2w_matrix = None

        if K is None:
            # VKITTI2 uses fixed camera intrinsics.
            # Reference: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
            # Fixed focal length 725, resolution 1242x375.
            K = torch.tensor(
                [
                    [725.0, 0.0, 621.0],
                    [0.0, 725.0, 187.5],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        if cam2w_matrix is None:
            cam2w_matrix = torch.eye(4, dtype=torch.float32)

        seq["camera_fields"].update({"camera", "cam2w"})
        camera = Pinhole(K=K[None, ...].clone())
        batch_camera = BatchCamera.from_camera(camera)
        seq["camera"] = batch_camera
        seq["cam2w"] = cam2w_matrix[None, ...]

        seq["metadata"] = metadata

        # Use ImageDataset preprocessing and postprocessing.
        from unidepth.datasets.image_dataset import ImageDataset
        results = ImageDataset.preprocess(self, results)

        if not self.test_mode:
            results = self.augment(results)

        results = ImageDataset.postprocess(self, results)

        return results

    def _load_depth_from_path(self, depth_path: str) -> np.ndarray:
        """Load a VKITTI2 depth map from disk and convert it into meters.

        Supported formats:
        - PNG (uint16): original VKITTI2 depth maps, in centimeters (cm)
        - NPY/NPZ: preprocessed depth maps, also in centimeters (cm)

        Returns: a depth array in meters (m), clipped to [min_depth, max_depth]
        """
        ext = os.path.splitext(depth_path)[1].lower()

        if ext == ".npy":
            depth = np.load(depth_path)
        elif ext == ".npz":
            data = np.load(depth_path)
            if "depth" in data:
                depth = data["depth"]
            else:
                key0 = list(data.keys())[0]
                depth = data[key0]
        elif ext == ".png":
            # VKITTI2 raw depth maps are 16-bit PNGs in centimeters.
            from PIL import Image
            depth_img = Image.open(depth_path)
            depth = np.array(depth_img, dtype=np.uint16)
        else:
            raise ValueError(f"Unsupported depth format: {depth_path} (ext: {ext})")

        # Handle multi-channel depth maps (take the first channel).
        if depth.ndim == 3:
            depth = depth[..., 0]

        # Convert to float32.
        depth = depth.astype(np.float32)

        # VKITTI2 depth maps are always in centimeters; convert to meters.
        depth = depth / 100.0

        # Clip depth to a reasonable range [min_depth, max_depth].
        # VKITTI2 has many extreme depths (>600m) that destabilize training.
        depth = np.clip(depth, self.min_depth, self.max_depth)

        return depth

    def log_load_dataset(self):
        """Log different info depending on mode."""
        from unidepth.utils import is_main_process
        if is_main_process():
            if self._use_manifest:
                # manifest mode: print the image count
                info = f"Loaded {self.__class__.__name__} with {len(self)} images."
            else:
                # HDF5 sequence mode: print both image and sequence counts
                info = f"Loaded {self.__class__.__name__} with {sum([len(x['image']) for x in self.sequences])} images in {len(self)} sequences."
            print(info)

    def __getitem__(self, idx):
        """Fetch a single dataset sample."""
        if self._use_manifest:
            # In manifest mode use ImageDataset's __getitem__.
            try:
                if isinstance(idx, (list, tuple)):
                    results = [self.get_single_item(i) for i in idx]
                else:
                    results = self.get_single_item(idx)
            except Exception as e:
                print(f"Error loading sequence {idx} for {self.__class__.__name__}: {e}")
                idx = np.random.randint(0, len(self.dataset))
                results = self[idx]
            return results
        else:
            # In HDF5 mode use SequenceDataset's __getitem__.
            return super().__getitem__(idx)

    def pre_pipeline(self, results):
        # In manifest mode call BaseDataset's pre_pipeline directly.
        if self._use_manifest:
            from unidepth.datasets.base_dataset import BaseDataset
            results = BaseDataset.pre_pipeline(self, results)
        else:
            # In HDF5 mode call SequenceDataset's pre_pipeline.
            results = super().pre_pipeline(results)

        num_items = self.num_frames * self.num_copies if hasattr(self, 'num_frames') else self.num_copies
        results["dense"] = [True] * num_items
        results["synthetic"] = [True] * num_items
        results["quality"] = [0] * num_items
        return results
