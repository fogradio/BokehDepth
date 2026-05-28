import json
import os
from typing import Any, Dict

import h5py
import numpy as np
import torch
from PIL import Image
from torchvision.io import read_image

from unidepth.datasets.image_dataset import ImageDataset
from unidepth.datasets.pipelines import AnnotationMask, KittiCrop
from unidepth.datasets.utils import DatasetFromList
from unidepth.utils import identity
from unidepth.utils.camera import BatchCamera, Pinhole


class KITTI(ImageDataset):
    CAM_INTRINSIC = {
        "2011_09_26": torch.tensor(
            [
                [7.215377e02, 0.000000e00, 6.095593e02, 4.485728e01],
                [0.000000e00, 7.215377e02, 1.728540e02, 2.163791e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 2.745884e-03],
            ]
        ),
        "2011_09_28": torch.tensor(
            [
                [7.070493e02, 0.000000e00, 6.040814e02, 4.575831e01],
                [0.000000e00, 7.070493e02, 1.805066e02, -3.454157e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 4.981016e-03],
            ]
        ),
        "2011_09_29": torch.tensor(
            [
                [7.183351e02, 0.000000e00, 6.003891e02, 4.450382e01],
                [0.000000e00, 7.183351e02, 1.815122e02, -5.951107e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 2.616315e-03],
            ]
        ),
        "2011_09_30": torch.tensor(
            [
                [7.070912e02, 0.000000e00, 6.018873e02, 4.688783e01],
                [0.000000e00, 7.070912e02, 1.831104e02, 1.178601e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 6.203223e-03],
            ]
        ),
        "2011_10_03": torch.tensor(
            [
                [7.188560e02, 0.000000e00, 6.071928e02, 4.538225e01],
                [0.000000e00, 7.188560e02, 1.852157e02, -1.130887e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 3.779761e-03],
            ]
        ),
    }
    min_depth = 0.05
    max_depth = 80.0
    depth_scale = 256.0
    log_mean = 2.5462
    log_std = 0.5871
    test_split = "kitti_eigen_test.txt"
    train_split = "kitti_eigen_train.txt"
    test_split_benchmark = "kitti_test.txt"
    hdf5_paths = ["kitti.hdf5"]

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
        **kwargs,
    ):
        self.defocus_stack_indices = kwargs.pop("defocus_stack_indices", None)
        if self.defocus_stack_indices is not None:
            self.defocus_stack_indices = [int(i) for i in self.defocus_stack_indices]

        self.manifest_path: str | None = kwargs.pop("manifest_path", None)
        if self.manifest_path:
            self.manifest_path = os.path.expanduser(self.manifest_path)
        self.invalid_depth_num = 0
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
        self.crop = crop
        self.cropper_base = KittiCrop(crop_size=(352, 1216))
        self.load_dataset()

    def get_intrinsics(self, idx, image_name):
        return self.CAM_INTRINSIC[image_name.split("/")[0]][:, :3].clone()

    def preprocess(self, results):
        results = self.replicate(results)
        for i, seq in enumerate(results["sequence_fields"]):
            self.resizer.ctx = None
            results[seq] = self.cropper_base(results[seq])
            results[seq] = self.resizer(results[seq])
            num_pts = torch.count_nonzero(results[seq]["depth"] > 0)
            if num_pts < 50:
                raise IndexError(f"Too few points in depth map ({num_pts})")

            for key in results[seq].get("image_fields", ["image"]):
                results[seq][key] = results[seq][key].to(torch.float32) / 255

        # update fields common in sequence
        for key in ["image_fields", "gt_fields", "mask_fields", "camera_fields"]:
            if key in results[(0, 0)]:
                results[key] = results[(0, 0)][key]
        results = self.pack_batch(results)
        return results

    def eval_mask(self, valid_mask, info={}):
        """Do grag_crop or eigen_crop for testing"""
        mask_height, mask_width = valid_mask.shape[-2:]
        eval_mask = torch.zeros_like(valid_mask)
        if "garg" in self.crop:
            eval_mask[
                ...,
                int(0.40810811 * mask_height) : int(0.99189189 * mask_height),
                int(0.03594771 * mask_width) : int(0.96405229 * mask_width),
            ] = 1
        elif "eigen" in self.crop:
            eval_mask[
                ...,
                int(0.3324324 * mask_height) : int(0.91351351 * mask_height),
                int(0.03594771 * mask_width) : int(0.96405229 * mask_width),
            ] = 1
        return torch.logical_and(valid_mask, eval_mask)

    def get_mapper(self):
        return {
            "image_filename": 0,
            "depth_filename": 1,
        }

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [False] * self.num_copies
        results["quality"] = [1] * self.num_copies
        return results

    def load_dataset(self):
        if self.manifest_path:
            manifest_path = self.manifest_path
            if not os.path.isabs(manifest_path):
                manifest_path = os.path.join(self.data_root, manifest_path)
            if not os.path.isfile(manifest_path):
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
        h5file.close()
        dataset = []
        for line in txt_string.split("\n"):
            image_filename = line.strip().split(" ")[0]
            depth_filename = line.strip().split(" ")[1]
            if depth_filename == "None":
                self.invalid_depth_num += 1
                continue
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
        samples: list[Dict[str, Any]] = []
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                image_path = payload.get("ref")
                # Prefer source_depth (original depth path); fall back to depth if missing.
                depth_path = payload.get("source_depth") or payload.get("depth")
                if image_path is None or depth_path is None:
                    continue

                scene = payload.get("scene") or ""
                date_key = scene.split("/")[0] if scene else ""
                if date_key not in self.CAM_INTRINSIC:
                    # fallback: try to infer from path
                    parts = image_path.split("/")
                    date_from_path = next(
                        (part for part in parts if part in self.CAM_INTRINSIC), None
                    )
                    if date_from_path:
                        date_key = date_from_path
                if date_key not in self.CAM_INTRINSIC:
                    raise KeyError(
                        f"Unable to infer KITTI intrinsic key for manifest entry: {payload}"
                    )
                K = self.CAM_INTRINSIC[date_key][:, :3].clone()

                image_abs = os.path.expanduser(image_path)
                depth_abs = os.path.expanduser(depth_path)
                if not os.path.isabs(image_abs):
                    image_abs = os.path.join(self.data_root, image_abs)

                # Handle the .npz[index] depth-path format.
                depth_index = None
                if "[" in depth_abs and depth_abs.endswith("]"):
                    # Parse the "path.npz[123]" format.
                    npz_path, index_str = depth_abs.rsplit("[", 1)
                    depth_index = int(index_str.rstrip("]"))
                    depth_file_to_check = npz_path
                else:
                    depth_file_to_check = depth_abs

                if not os.path.isabs(depth_file_to_check):
                    depth_file_to_check = os.path.join(self.data_root, depth_file_to_check)
                    if depth_index is not None:
                        depth_abs = f"{depth_file_to_check}[{depth_index}]"
                    else:
                        depth_abs = depth_file_to_check

                if not os.path.isfile(image_abs):
                    raise FileNotFoundError(f"Image not found: {image_abs}")
                if not os.path.isfile(depth_file_to_check):
                    raise FileNotFoundError(f"Depth file not found: {depth_file_to_check}")

                # Process the defocus stack.
                stack_paths = payload.get("stack", [])
                k_values = payload.get("k", [])

                # Validate the defocus-stack paths.
                valid_stack_paths = []
                if stack_paths:
                    for stack_path in stack_paths:
                        stack_path = os.path.expanduser(stack_path)
                        if not os.path.isabs(stack_path):
                            stack_path = os.path.join(self.data_root, stack_path)
                        if os.path.isfile(stack_path):
                            valid_stack_paths.append(stack_path)

                # Pick particular stack indices based on config.
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

                samples.append(
                    {
                        "image_path": image_abs,
                        "depth_path": depth_abs,
                        "depth_index": depth_index,
                        "stack_paths": valid_stack_paths,
                        "k_values": k_values,
                        "K": K,
                    }
                )

        if not samples:
            raise RuntimeError(f"No valid KITTI samples parsed from manifest {manifest_path}")

        self.dataset = DatasetFromList(samples)
        self.use_manifest = True
        self.log_load_dataset()

    def get_single_item(self, idx, sample=None, mapper=None):
        if not getattr(self, "use_manifest", False):
            return super().get_single_item(idx, sample=sample, mapper=mapper)

        sample = self.dataset[idx] if sample is None else sample
        image = Image.open(sample["image_path"]).convert("RGB")
        image_tensor = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1)

        # Load the depth map based on its format.
        depth_path = sample["depth_path"]
        depth_index = sample.get("depth_index")

        if depth_index is not None:
            # Load by index from a .npz file.
            npz_path = depth_path.rsplit("[", 1)[0]
            with np.load(npz_path, allow_pickle=True) as npz_data:
                # .npz files usually contain a 'data' key or are an array list.
                if 'data' in npz_data:
                    depth_np = npz_data['data'][depth_index].astype(np.float32, copy=False)
                else:
                    # Get the data under the first key.
                    key = list(npz_data.keys())[0]
                    depth_np = npz_data[key][depth_index].astype(np.float32, copy=False)
        else:
            # Load a .npy file directly.
            depth_np = np.load(depth_path).astype(np.float32, copy=False)

        depth_tensor = torch.from_numpy(depth_np)

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
        seq["filename"] = sample["image_path"]
        seq["image_fields"].add("image")
        seq["gt_fields"].add("depth")
        seq["camera_fields"].update({"camera", "cam2w", "K"})
        seq["image"] = image_tensor.unsqueeze(0)
        seq["image_ori_shape"] = image_tensor.shape[-2:]
        seq["depth"] = depth_tensor.view(1, 1, *depth_tensor.shape)
        seq["depth_ori_shape"] = depth_tensor.shape

        # Load defocus-stack images.
        stack_paths = sample.get("stack_paths", [])
        k_values = sample.get("k_values", [])
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

        camera = Pinhole(K=sample["K"][None, ...])
        seq["camera"] = BatchCamera.from_camera(camera)
        seq["cam2w"] = torch.eye(4, dtype=torch.float32)[None, ...]
        seq["K"] = sample["K"][None, ...]

        results = self.preprocess(results)
        if not self.test_mode:
            results = self.augment(results)
        results = self.postprocess(results)
        return results


class KITTIBenchmark(ImageDataset):
    min_depth = 0.05
    max_depth = 80.0
    depth_scale = 256.0
    test_split = "test_split.txt"
    train_split = "val_split.txt"
    intrinsics_file = "intrinsics.json"
    hdf5_paths = ["kitti_benchmark.hdf5"]

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
        **kwargs,
    ):
        super().__init__(
            image_shape=image_shape,
            split_file=split_file,
            test_mode=test_mode,
            benchmark=True,
            normalize=normalize,
            augmentations_db=augmentations_db,
            resize_method=resize_method,
            mini=mini,
            **kwargs,
        )
        self.test_mode = test_mode

        self.crop = crop

        self.masker = AnnotationMask(
            min_value=self.min_depth,
            max_value=self.max_depth if test_mode else None,
            custom_fn=lambda x, *args, **kwargs: x,
        )
        self.collecter = Collect(keys=["image_fields", "mask_fields", "gt_fields"])
        self.load_dataset()

    def load_dataset(self):
        h5file = h5py.File(
            os.path.join(self.data_root, self.hdf5_path),
            "r",
            libver="latest",
            swmr=True,
        )
        txt_file = np.array(self.h5file[self.split_file])
        txt_string = txt_file.tostring().decode("ascii")[:-1]  # correct the -1
        intrinsics = np.array(h5file[self.intrinsics_file]).tostring().decode("ascii")
        intrinsics = json.loads(intrinsics)
        h5file.close()
        dataset = []
        for line in txt_string.split("\n"):
            image_filename, depth_filename = line.strip().split(" ")
            intrinsics = torch.tensor(
                intrinsics[os.path.join(*image_filename.split("/")[:2])]
            ).squeeze()[:, :3]
            sample = {
                "image_filename": image_filename,
                "depth_filename": depth_filename,
                "K": intrinsics,
            }
            dataset.append(sample)

        self.dataset = DatasetFromList(dataset)

        self.log_load_dataset()
