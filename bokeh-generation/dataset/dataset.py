"""Training-time dataset wrappers and collate helpers.

Supported sample formats (auto-detected via JSONL fields):

* **ITW** (in-the-wild Flickr): ``image_itw_path`` / ``input_image_path`` +
  optional ``fg_mask_path`` + ``depth_map_path``. Used for both T2I (with
  on-the-fly BokehMe synthesis) and I2I (when ``target_image_path`` is set).
* **BLB / Aperture / DPDD / EBB-aligned**: an ``input_image_path`` plus
  ``target_image_path`` for the pre-rendered bokeh image, used for I2I.

All paths in the JSONL are expected to be absolute.
"""

import json
import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import make_grid

from constants import FLUX_SIZE_PRESETS
from dataset.bokehme.demo import pipeline


K_MIN = 1.0
K_MAX = 30.0


class FlickrInTheWildDataset(Dataset):
    """Unified training dataset wrapping ITW and BLB-style JSONLs.

    Parameters
    ----------
    dataset_file
        Path to a JSONL whose schema matches one of the supported families
        (see module docstring).
    camera_anns
        List of camera-annotation keys to expose as conditioning, e.g.
        ``["dof-cond"]``.
    tokenizer
        Optional HuggingFace tokenizer; if ``None`` no text ids are produced.
    size
        Target square size; set to ``None`` to keep the original resolution
        (variable-resolution training).
    uncond_prob
        Probability of dropping the caption to the empty string (used by
        classifier-free guidance).
    is_main_process
        For nicer logging in the distributed setting.
    synthetic_pairing
        Restrict to samples flagged ``suitable_for_synthetic`` and emit
        un-normalised camera values (used by the synthetic-pairs collate fn).
    horizontal_flip
        Apply random horizontal flips jointly to image / fg mask / depth.
    filter_recency
        ITW-only filter: keep only photos with ``photo_id`` above an upper
        threshold (newer uploads tend to be higher quality).
    """

    def __init__(
        self,
        dataset_file,
        camera_anns,
        tokenizer=None,
        size=512,
        uncond_prob=0.1,
        is_main_process=False,
        synthetic_pairing=False,
        horizontal_flip=True,
        filter_recency=False,
    ):
        super().__init__()
        self.size = size
        self.tokenizer = tokenizer
        self.dataset_file = dataset_file
        self.camera_anns = camera_anns
        self.uncond_prob = uncond_prob
        self.is_main_process = is_main_process
        self.synthetic_pairing = synthetic_pairing
        self.horizontal_flip = horizontal_flip

        self.photos = self.init_photos(dataset_file, synthetic_pairing, filter_recency)
        # Per-sample, per-annotation list of (possibly normalised) scalar values
        self.camera_anns = [
            self.prepare_camera_parameter(key=ann, synthetic_pairing=synthetic_pairing)
            for ann in self.camera_anns
        ]
        self.camera_anns = [list(item) for item in zip(*self.camera_anns)]

    # ------------------------------------------------------------------ init
    def init_photos(self, dataset_file, synthetic_pairing=False, filter_recency=False):
        with open(dataset_file) as f:
            samples = [json.loads(line) for line in f]
        if synthetic_pairing:
            samples = [s for s in samples if s.get("suitable_for_synthetic", False)]
        if filter_recency:
            samples = [s for s in samples if int(s.get("photo_id", 0)) > 20000000000]
        return samples

    def prepare_camera_parameter(self, key="dof-cond", synthetic_pairing=False):
        """Collect the raw values for a single camera-annotation key.

        Looks up ``key`` in two possible locations to keep compatibility
        with all supported JSONL families:

        1. ``photo["camera_anns"][key]`` (or ``key + "-crop"`` for cropped variants)
        2. top-level ``photo[key]``

        Returns the raw list when ``synthetic_pairing`` is True; otherwise
        returns a min-max normalised version in [0, 1].
        """
        raw_vals = []
        missing_logged = 0

        for photo in self.photos:
            param_value = None

            # 1) ITW-style camera_anns dict
            if "camera_anns" in photo and isinstance(photo["camera_anns"], dict):
                if self.size != 1024 and f"{key}-crop" in photo["camera_anns"]:
                    param_value = photo["camera_anns"][f"{key}-crop"]
                elif key in photo["camera_anns"]:
                    param_value = photo["camera_anns"][key]

            # 2) Top-level field fallback
            if param_value is None and key in photo:
                param_value = photo[key]

            if param_value is not None:
                raw_vals.append(param_value)
            elif missing_logged < 5:
                photo_id = photo.get("photo_id", photo.get("combination_id", "unknown"))
                print(
                    f"[WARN] missing camera annotation '{key}' for sample {photo_id} "
                    f"(dataset_type={photo.get('dataset_type', 'unknown')}); skipping"
                )
                missing_logged += 1

        if not raw_vals:
            raise ValueError(
                f"No valid value for camera annotation '{key}' across the dataset. "
                "Check the JSONL schema or the requested annotation name."
            )

        if synthetic_pairing:
            return raw_vals
        vmin, vmax = min(raw_vals), max(raw_vals)
        return [(v - vmin) / (vmax - vmin + 1e-8) for v in raw_vals]

    # ------------------------------------------------------------ geometry
    def _get_resize_crop_params(self, width, height, target_size):
        """Compute centre-crop parameters for a square target_size."""
        if target_size is None:
            return {"ratio": 1.0, "new_w": width, "new_h": height, "left": 0, "top": 0}

        if width == target_size and height == target_size:
            return {"ratio": 1.0, "new_w": width, "new_h": height, "left": 0, "top": 0}
        ratio = target_size / min(width, height)
        new_w = int(np.ceil(width * ratio))
        new_h = int(np.ceil(height * ratio))
        left = int(round((new_w - target_size) / 2))
        top = int(round((new_h - target_size) / 2))
        return {"ratio": ratio, "new_w": new_w, "new_h": new_h, "left": left, "top": top}

    def _resize_and_crop_np(self, img_np, target_size=512, params=None, interpolation=cv2.INTER_AREA):
        """Resize and centre-crop ``img_np`` to ``target_size``.

        Two code paths:
        - If ``params`` carries pre-computed normalised crop info
          (``left_norm`` etc.), reuse it so that input image, fg-mask, depth
          map and the I2I target are all cropped consistently.
        - Otherwise, compute a fresh centre crop on this modality alone.
        """
        h, w = img_np.shape[:2]

        # Variable-resolution mode: keep the original image untouched
        if target_size is None:
            return img_np, {"ratio": 1.0, "new_w": w, "new_h": h, "left": 0, "top": 0}

        if params is not None and "left_norm" in params:
            left = int(round(params["left_norm"] * w))
            top = int(round(params["top_norm"] * h))
            crop_width = int(round(params["crop_width_norm"] * w))
            crop_height = int(round(params["crop_height_norm"] * h))
            cropped = img_np[top : top + crop_height, left : left + crop_width]
            resized = cv2.resize(cropped, (target_size, target_size), interpolation=interpolation)
            return resized, params

        params_new = self._get_resize_crop_params(w, h, target_size)
        # cv2.resize expects size as (width, height)
        resized = cv2.resize(img_np, (params_new["new_w"], params_new["new_h"]), interpolation=interpolation)
        cropped = resized[
            params_new["top"] : params_new["top"] + target_size,
            params_new["left"] : params_new["left"] + target_size,
        ]
        return cropped, params_new

    # ------------------------------------------------------------ __getitem__
    def __getitem__(self, idx):
        photo = self.photos[idx]

        # ------ Caption (with classifier-free dropout) ------
        if "captions" in photo and photo["captions"]:
            caption = random.choice(photo["captions"]) if random.random() > self.uncond_prob else ""
        elif "i2i_prompt_template" in photo:
            caption = photo["i2i_prompt_template"] if random.random() > self.uncond_prob else ""
        elif "scene_description" in photo:
            caption = photo["scene_description"] if random.random() > self.uncond_prob else ""
        else:
            caption = ""

        cam_ann = self.camera_anns[idx]

        # Decide once whether to flip; reuse for image, mask, depth, and the I2I target
        flip_flag = self.horizontal_flip and (random.random() < 0.5)

        # ------ Main image ------
        # We always load the un-blurred input. If the sample carries crop_info
        # with explicit crop_params, we honour them; otherwise we compute a
        # fresh centre crop. This keeps the input aligned with the target
        # bokeh image for I2I training.
        crop_params_candidate = None
        if "crop_info" in photo and "flux" in photo["crop_info"]:
            crop_params_candidate = photo["crop_info"]["flux"].get("crop_params")

        # Resolve the path to the input image across dataset families
        if photo.get("image_itw_path"):
            image_path = photo["image_itw_path"]
        elif photo.get("input_image_path"):
            image_path = photo["input_image_path"]
        else:
            raise ValueError(f"No valid input image field in sample keys: {list(photo.keys())}")

        image_np = cv2.imread(image_path)
        if image_np is None:
            raise ValueError(f"Failed to read image: {image_path}")

        if crop_params_candidate is not None:
            image_np, params = self._resize_and_crop_np(image_np, self.size, crop_params_candidate)
        else:
            image_np, params = self._resize_and_crop_np(image_np, self.size)

        if flip_flag:
            image_np = cv2.flip(image_np, 1)

        height, width = image_np.shape[:2]
        # FLUX uses a VAE with scale factor 8 plus a 2x pack/unpack downstream, so
        # both spatial dims must be divisible by 16 (skip the check in variable-resolution mode).
        if self.size is not None:
            assert (
                height % 16 == 0 and width % 16 == 0
            ), f"Image shape {image_np.shape} is not divisible by 16 required by FLUX."

        # Convert BGR -> RGB, normalise to [-1, 1]
        image_np = cv2.cvtColor(np.array(image_np, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        image_torch = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(
            torch.from_numpy(image_np).permute(2, 0, 1).float().div(255.0)
        )

        # ------ Foreground mask ------
        fg_mask_path = photo.get("fg_mask_path", "")
        if fg_mask_path and fg_mask_path.strip():
            fg_mask_np = cv2.imread(fg_mask_path, cv2.IMREAD_GRAYSCALE)
            if fg_mask_np is None:
                fg_mask_np = np.ones((image_np.shape[0], image_np.shape[1]), dtype=np.uint8) * 255
        else:
            fg_mask_np = np.ones((image_np.shape[0], image_np.shape[1]), dtype=np.uint8) * 255

        fg_mask_np, _ = self._resize_and_crop_np(fg_mask_np, self.size, params, interpolation=cv2.INTER_NEAREST)
        if flip_flag:
            fg_mask_np = cv2.flip(fg_mask_np, 1)
        fg_mask_torch = torch.from_numpy(fg_mask_np).unsqueeze(0).float().div(255.0).round()

        # ------ Depth map (optional; falls back to zeros) ------
        depth_path = photo.get("depth_map_path")

        try:
            depth_map = None
            if depth_path and isinstance(depth_path, str) and depth_path.strip() and os.path.exists(depth_path):
                if depth_path.endswith(".npz"):
                    depth_map = np.load(depth_path)["depth"].astype(np.float32)
                elif depth_path.endswith(".npy"):
                    depth_map = np.load(depth_path).astype(np.float32)

            if depth_map is None:
                # Zero placeholder when depth is unavailable (common for pre-rendered I2I samples)
                depth_map = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
            depth_map, _ = self._resize_and_crop_np(depth_map, self.size, params)
            if flip_flag:
                depth_map = cv2.flip(depth_map, 1)
            depth_map_torch = torch.from_numpy(depth_map).unsqueeze(0).float()
        except Exception:
            depth_map = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
            depth_map, _ = self._resize_and_crop_np(depth_map, self.size, params)
            if flip_flag:
                depth_map = cv2.flip(depth_map, 1)
            depth_map_torch = torch.from_numpy(depth_map).unsqueeze(0).float()

        # ------ Tokenize caption ------
        if self.tokenizer is not None:
            inputs = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            text_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        else:
            text_ids = torch.zeros(1, 0)
            attention_mask = torch.zeros(1, 0)

        # ------ Normalised crop params (so the I2I target can be cropped identically) ------
        crop_params_norm = None
        if params is not None:
            if "left_norm" in params:
                crop_params_norm = params
            elif crop_params_candidate is not None:
                crop_params_norm = crop_params_candidate
            elif "ratio" in params:
                if self.size is not None:
                    crop_width_norm = self.size / params["new_w"] if params["new_w"] > 0 else 1
                    crop_height_norm = self.size / params["new_h"] if params["new_h"] > 0 else 1
                else:
                    crop_width_norm = 1
                    crop_height_norm = 1
                crop_params_norm = {
                    "ratio": params["ratio"],
                    "left_norm": params["left"] / params["new_w"] if params["new_w"] > 0 else 0,
                    "top_norm": params["top"] / params["new_h"] if params["new_h"] > 0 else 0,
                    "crop_width_norm": crop_width_norm,
                    "crop_height_norm": crop_height_norm,
                }

        return {
            "image": image_torch,               # in [-1, 1]
            "image_np": image_np,               # for downstream BokehMe pipeline
            "fg_mask": fg_mask_torch,           # (1, H, W) in [0, 1]
            "fg_mask_np": fg_mask_np,           # for downstream BokehMe pipeline
            "depth_map": depth_map_torch,       # (1, H, W)
            "camera_ann": cam_ann,
            "caption": caption,
            "text_input_ids": text_ids,
            "text_attention_mask": attention_mask,
            # ITW-specific metadata pass-through
            "suitable_for_synthetic": photo.get("suitable_for_synthetic", False),
            "task_type": photo.get("task_type", "adjust_bokeh"),
            "disp_focus": photo.get("disp_focus", 0.5),
            # I2I-specific pass-through
            "target_image_path": photo.get("target_image_path", None),
            "dataset_type": "blb" if photo.get("blb_metadata") else None,
            "foreground_clear": photo.get("foreground_clear", True),
            # Geometry the I2I target needs to mirror
            "flip_flag": flip_flag,
            "crop_params_norm": crop_params_norm,
        }

    def __len__(self):
        return len(self.photos)


# ---------------------------------------------------------------- collates
def collate_fn(batch, uncond_prob=0.1):
    """Plain T2I collate (no synthetic pairing)."""
    for ex in batch:
        if random.random() < uncond_prob:
            ex["camera_ann"] = torch.full_like(torch.tensor(ex["camera_ann"], dtype=torch.float32), -1)
        else:
            ex["camera_ann"] = torch.tensor(ex["camera_ann"], dtype=torch.float32)
    return {
        "images": torch.stack([ex["image"] for ex in batch]),
        "captions": [ex["caption"] for ex in batch],
        "text_input_ids": torch.cat([ex["text_input_ids"] for ex in batch], dim=0),
        "text_attention_mask": torch.cat([ex["text_attention_mask"] for ex in batch], dim=0),
        "fg_masks": torch.stack([ex["fg_mask"] for ex in batch]),
        "depth_maps": torch.stack([ex["depth_map"] for ex in batch]),
        "camera_anns": torch.tensor([ex["camera_ann"] for ex in batch], dtype=torch.float32),
        "is_synthetic": False,
        "target_image_path": [ex.get("target_image_path", None) for ex in batch],
        "dataset_type": [ex.get("dataset_type", None) for ex in batch],
        "foreground_clear": [ex.get("foreground_clear", True) for ex in batch],
    }


def collate_fn_synthetic_pairs(
    batch,
    synth_sample,
    classical_renderer,
    arnet,
    iunet,
    device,
    uncond_prob=0.1,
    vis=False,
    is_main_process=False,
    swap_prob=0.2,
    K_min=K_MIN,
    K_max=K_MAX,
):
    """Build synthetic bokeh pairs on the fly using BokehMe (T2I path).

    For each "real" sample in the batch we generate ``synth_sample`` extra
    bokeh versions at random K values, so the model sees both the in-focus
    image and several blurred renditions with different conditioning.
    """
    final_items = []
    batch_size = (len(batch) * (synth_sample + 1))
    synth_sample -= max(batch_size // 10 - len(batch), 0)

    for real_item in batch:
        # A) Real (in-focus) entry
        real_dict = {
            "image": real_item["image"],
            "fg_mask": real_item["fg_mask"],
            "depth_map": real_item["depth_map"],
            "defocus_map": torch.zeros_like(real_item["depth_map"]),
            "camera_ann": torch.tensor([real_item["camera_ann"][0] / K_MAX], dtype=torch.float32),
            "caption": real_item["caption"],
            "text_input_ids": real_item["text_input_ids"],
            "text_attention_mask": real_item["text_attention_mask"],
        }
        final_items.append(real_dict)

        # B) Generate ``synth_sample`` synthetic bokeh versions
        for _ in range(synth_sample):
            with torch.no_grad():
                bokeh_pred, defocus_map, K, flat_fg = add_bokeh(
                    image_np=real_item["image_np"],
                    fg_mask_np=real_item["fg_mask_np"],
                    depth_map=real_item["depth_map"].squeeze(0).numpy(),
                    classical_renderer=classical_renderer,
                    arnet=arnet,
                    iunet=iunet,
                    device=device,
                    K_min=K_min,
                    K_max=K_max,
                    is_main_process=is_main_process,
                )

            synth_dict = {
                "image": bokeh_pred,
                "fg_mask": real_item["fg_mask"],
                "depth_map": real_item["depth_map"],
                "defocus_map": defocus_map,
                "camera_ann": torch.tensor([K / K_MAX], dtype=torch.float32),
                "caption": real_item["caption"],
                "text_input_ids": real_item["text_input_ids"],
                "text_attention_mask": real_item["text_attention_mask"],
                "flat_fg": flat_fg,
            }
            final_items.append(synth_dict)

        # Pad to the requested batch size by repeating the real entry
        while len(final_items) < batch_size:
            final_items.insert(0, real_dict)

    swap_id = random.randint(0, batch_size - 1)
    batch_swap_ids = [swap_id if random.random() < swap_prob else i for i in range(batch_size)]

    # Classifier-free dropout for the camera annotation
    for i in range(len(final_items)):
        if random.random() < uncond_prob:
            final_items[i]["camera_ann"] = torch.full_like(final_items[i]["camera_ann"], -1)

    # Optional visualisation grid for debugging
    if vis and is_main_process:
        vis_images = [(ex["image"] + 1) / 2 for ex in final_items]
        vis_depths = []
        for ex in final_items:
            depth = ex["depth_map"][0]
            inverse_depth = 1 / depth
            max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
            min_invdepth_vizu = max(1 / 250, inverse_depth.min())
            inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu)
            cmap = plt.get_cmap("turbo")
            colored_depth = torch.from_numpy(cmap(inverse_depth_normalized.cpu().numpy())[:, :, :3]).permute(2, 0, 1)
            vis_depths.append(colored_depth)
        vis_masks = [ex["fg_mask"].repeat(3, 1, 1) for ex in final_items]
        vis_combined = vis_images + vis_depths + vis_masks

        grid = make_grid(vis_combined, nrow=synth_sample + 1, padding=2, normalize=False)
        grid_pil = TF.to_pil_image(grid)
        os.makedirs("debug_vis", exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        grid_pil.save(f"debug_vis/batch_{timestamp}.jpg")

    return {
        "images": torch.stack([ex["image"] for ex in final_items]),
        "captions": [ex["caption"] for ex in final_items],
        "text_input_ids": torch.cat([ex["text_input_ids"] for ex in final_items], dim=0),
        "text_attention_mask": torch.cat([ex["text_attention_mask"] for ex in final_items], dim=0),
        "fg_masks": torch.stack([ex["fg_mask"] for ex in final_items]),
        "depth_maps": torch.stack([ex["depth_map"] for ex in final_items]),
        "camera_anns": torch.stack([ex["camera_ann"] for ex in final_items]),
        "is_synthetic": True,
        # When no synthetic samples were generated, the last entry may be the real_dict (no flat_fg field)
        "flat_fg": final_items[-1].get("flat_fg", False),
        "batch_swap_ids": batch_swap_ids,
    }


def collate_fn_i2i(batch, uncond_prob=0.1):
    """I2I collate.

    - Drops samples that explicitly mark ``foreground_clear`` as False so the
      target image is guaranteed to keep the subject sharp.
    - Pass-through fields needed to locate the I2I target on the trainer side.
    """
    filtered = [ex for ex in batch if ex.get("foreground_clear", True)]
    if len(filtered) == 0:
        return {"skip_batch": True, "is_i2i": True}

    for ex in filtered:
        if random.random() < uncond_prob:
            ex["camera_ann"] = torch.full_like(torch.tensor(ex["camera_ann"], dtype=torch.float32), -1)
        else:
            ex["camera_ann"] = torch.tensor(ex["camera_ann"], dtype=torch.float32)

    return {
        "images": torch.stack([ex["image"] for ex in filtered]),
        "captions": [ex["caption"] for ex in filtered],
        "text_input_ids": torch.cat([ex["text_input_ids"] for ex in filtered], dim=0),
        "text_attention_mask": torch.cat([ex["text_attention_mask"] for ex in filtered], dim=0),
        "fg_masks": torch.stack([ex["fg_mask"] for ex in filtered]),
        "depth_maps": torch.stack([ex["depth_map"] for ex in filtered]),
        "camera_anns": torch.tensor([ex["camera_ann"] for ex in filtered], dtype=torch.float32),
        "is_synthetic": False,
        "target_image_path": [ex.get("target_image_path", None) for ex in filtered],
        "dataset_type": [ex.get("dataset_type", None) for ex in filtered],
        "foreground_clear": [ex.get("foreground_clear", True) for ex in filtered],
        # Raw ITW input arrays kept for downstream BokehMe usage
        "image_np_list": [ex.get("image_np") for ex in filtered],
        "fg_mask_np_list": [ex.get("fg_mask_np") for ex in filtered],
        # Geometry the target image must mirror to stay pixel-aligned
        "flip_flags": [ex.get("flip_flag", False) for ex in filtered],
        "crop_params_norm": [ex.get("crop_params_norm", None) for ex in filtered],
        "is_i2i": True,
    }


def add_bokeh(
    image_np,
    fg_mask_np,
    depth_map,
    classical_renderer,
    arnet,
    iunet,
    device,
    K_min,
    K_max,
    gamma=2.2,
    gamma_min=1.0,
    gamma_max=5.0,
    defocus_scale=10.0,
    is_main_process=False,
):
    """Render a synthetic bokeh image with BokehMe.

    Inputs (all already share the same geometric augmentation):
    - ``image_np``: [H, W, 3] uint8 in 0..255
    - ``fg_mask_np``: [H, W] uint8 in 0..255
    - ``depth_map``: [H, W] float depth in metric units

    Returns ``(bokeh_pred_torch, defocus_torch, K, flat_fg_flag)``.
    """
    # Robust disparity from depth: guard zeros / NaNs / infs and handle sparse depth gracefully
    depth_np = np.asarray(depth_map, dtype=np.float32)
    valid_mask = np.isfinite(depth_np) & (depth_np > 1e-6)
    if not np.any(valid_mask):
        # No valid depth: fall back to a neutral 0.5 disparity
        disp = np.full_like(depth_np, 0.5, dtype=np.float32)
    else:
        inv = np.zeros_like(depth_np, dtype=np.float32)
        np.divide(1.0, depth_np, out=inv, where=valid_mask)
        vmin = float(np.min(inv[valid_mask]))
        vmax = float(np.max(inv[valid_mask]))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-8:
            disp = np.full_like(depth_np, 0.5, dtype=np.float32)
        else:
            disp = (inv - vmin) / (vmax - vmin + 1e-8)
            # Fill invalid pixels with the median of valid disparities for stability
            disp_med = float(np.median(disp[valid_mask]))
            disp[~valid_mask] = disp_med

    flat_fg = False
    fg_mask_condition = (fg_mask_np > 127) & valid_mask
    fg_depth = depth_np[fg_mask_condition]
    if fg_depth.size == 0:
        disp_focus = float(np.median(disp))
    else:
        depth_threshold = np.percentile(fg_depth, 96)
        mask_closest = (depth_map <= depth_threshold) & fg_mask_condition

        fg_depth_closest = depth_map[mask_closest]
        if fg_depth_closest.size == 0:
            disp_focus = float(np.median(disp))
        else:
            fg_disp_closest = disp[mask_closest]
            if fg_disp_closest.size == 0:
                disp_focus = float(np.median(disp))
            else:
                perc50 = np.percentile(fg_disp_closest, 50)
                filtered = fg_disp_closest[fg_disp_closest > perc50]
                if filtered.size == 0:
                    disp_focus = float(np.median(fg_disp_closest))
                else:
                    disp_focus = float(np.median(filtered))

            range_depth = np.max(fg_depth_closest) - np.min(fg_depth_closest)
            THRESHOLD = 10.0
            if range_depth <= THRESHOLD:
                # Re-normalise using foreground-only disparity extremes (not full-image) to avoid biasing the focal plane
                original_disp = np.zeros_like(depth_np, dtype=np.float32)
                np.divide(1.0, depth_np, out=original_disp, where=valid_mask)
                fg_disp = original_disp[fg_mask_condition]
                disp_threshold = 1.0 / depth_threshold
                disp_focus = (disp_threshold - fg_disp.min()) / (fg_disp.max() - fg_disp.min() + 1e-8)
                disp_focus = np.clip(disp_focus, 0.0, 1.0)
                disp[mask_closest] = disp_focus
                flat_fg = True

    K = float(random.uniform(float(K_min), float(K_max)))
    defocus = K * (disp - disp_focus) / max(defocus_scale, 1e-6)
    defocus = np.nan_to_num(defocus, nan=0.0, posinf=0.0, neginf=0.0)

    image_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_t = image_t.unsqueeze(0).to(device)
    defocus_t = torch.from_numpy(defocus.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)

    with torch.no_grad():
        bokeh_pred, _, _, _ = pipeline(
            classical_renderer,
            arnet,
            iunet,
            image_t,
            defocus_t,
            gamma=gamma,
            defocus_scale=defocus_scale,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
    bokeh_np = bokeh_pred[0].cpu().permute(1, 2, 0).numpy()

    bokeh_t = torch.from_numpy(bokeh_np).permute(2, 0, 1).float()
    bokeh_t = (bokeh_t * 2.0) - 1.0
    return bokeh_t, defocus_t, K, flat_fg


# Reference for FLUX-supported resolutions; exported in case callers want it.
__all__ = [
    "K_MIN",
    "K_MAX",
    "FLUX_SIZE_PRESETS",
    "FlickrInTheWildDataset",
    "collate_fn",
    "collate_fn_synthetic_pairs",
    "collate_fn_i2i",
    "add_bokeh",
]
