"""Stage-1 LoRA training script for the bokeh-generation FLUX adapter.

This is the I2I (image-to-image) training entry point that produces the
``BokehFluxControlAdapter`` LoRA released alongside the inference code in
``gen_bokeh_stack.py``.

High-level overview
-------------------
- Base model: FLUX.1-Kontext-dev. The script trains only the adapter LoRA
  weights (plus optional Q/K unfreezing) on top of a fully frozen Kontext
  pipeline (transformer + VAE + dual text encoders).
- Two batch modes mixed per epoch:
  * **T2I**: pure text-to-image. For each step we draw either a *real*
    in-focus sample or an on-the-fly *synthetic* bokeh pair rendered by
    BokehMe (ARNet + IUNet + classical scatter).
  * **I2I**: image-to-image. The input is the all-in-focus image, the
    target is a pre-rendered (or freshly BokehMe-synthesised) bokeh image.
- Optional features (CLI flags): variable-resolution training, smart
  checkpointing (latest + best with crash-resilient rollback), gradient
  checkpointing, memory-efficient attention, and OOM-fallback wrappers
  around VAE encode and the Transformer forward pass.

Required data inputs (all CLI-configurable):
- ``--itw_jsonl``       : in-the-wild Flickr-style JSONL used for T2I /
                          synthetic pairs and as an additional I2I source.
- ``--i2i_jsonl``       : repeatable; pre-rendered I2I JSONLs (e.g. BLB,
                          DPDD, Aperture, EBB-aligned). Provide one flag
                          per JSONL to mix multiple sources.
- ``--post_bokeme_jsonl`` (optional, with ``--post_bokeme_syn`` for
                          *only* online BokehMe synthesis, or with
                          ``--include_post_bokeme_syn`` to add it to a
                          mixed run).
- ``--arnet_ckpt`` / ``--iunet_ckpt`` : BokehMe checkpoints used by the
                          synthetic data path (default:
                          ``dataset/bokehme/checkpoints/{arnet,iunet}.pth``).
"""

import argparse
import copy
import itertools
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import prodigyopt
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import (
    FlowMatchEulerDiscreteScheduler,  # noqa: F401  (kept for downstream use)
    FluxKontextPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from lion_pytorch import Lion
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from constants import FLUX_TRANSFORMER_BLOCKS
from dataset.bokehme.classical_renderer.scatter import ModuleRenderScatter
from dataset.bokehme.neural_renderer import ARNet, IUNet
from dataset.dataset import (
    FlickrInTheWildDataset,
    add_bokeh,
    collate_fn,
    collate_fn_i2i,
    collate_fn_synthetic_pairs,
)
from model.bokeh_adapter_flux import BokehFluxControlAdapter
from utils import compress_block_ids, parse_block_ids

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train the BokehFluxControlAdapter LoRA in I2I mode")

    # Base model + adapter selection
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--block_ids", type=parse_block_ids, default="0-56")
    parser.add_argument("--camera_anns", type=lambda x: x.split(","), default="dof-cond")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to a previous accelerate state dir to resume from.")
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)

    # Dataset paths (no hardcoded defaults)
    parser.add_argument("--itw_jsonl", type=str, default=None,
                        help="In-the-wild Flickr-style JSONL; used for T2I/synthetic pairs and as an extra I2I source.")
    parser.add_argument("--i2i_jsonl", type=str, action="append", default=None,
                        help="Repeatable: paths to pre-rendered I2I JSONLs. Combine multiple sources by passing the flag again.")
    parser.add_argument("--post_bokeme_jsonl", type=str, default=None,
                        help="JSONL of all-in-focus + depth pairs used to synthesise targets online with BokehMe.")

    # BokehMe (synthetic pipeline) checkpoints
    parser.add_argument("--arnet_ckpt", type=str, default="dataset/bokehme/checkpoints/arnet.pth")
    parser.add_argument("--iunet_ckpt", type=str, default="dataset/bokehme/checkpoints/iunet.pth")

    # Data loading
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--real_data_ratio", type=float, default=0.5)
    parser.add_argument("--unfreeze_q", action="store_true")
    parser.add_argument("--unfreeze_k", action="store_true")
    parser.add_argument("--K_min", type=float, default=1.0)
    parser.add_argument("--K_max", type=float, default=30.0)
    parser.add_argument("--swap_prob", type=float, default=1.0,
                        help="T2I grounded-attention swap probability (>0 enables the swap path).")
    parser.add_argument("--enable_grounded_attention", action="store_true", default=True,
                        help="Enable grounded attention in T2I (force-disabled by the I2I loop).")

    # Training schedule
    parser.add_argument("--max_train_epochs", type=int, default=30)
    parser.add_argument("--noise_offset", type=float, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--weighting_scheme", type=str, default="none")
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "lion", "prodigy"])
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)

    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_resume_id", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every_n_epochs", type=int, default=1)

    # I2I / T2I mixing
    parser.add_argument("--i2i_ratio", type=float, default=0.5,
                        help="Fraction of every epoch reserved for I2I steps (0..1).")
    parser.add_argument("--i2i_prompt_keep_ratio", type=float, default=0.35,
                        help="Probability of keeping the original caption in I2I mode; else use a template.")
    parser.add_argument("--i2i_horizontal_flip", action="store_true", default=True,
                        help="Apply random horizontal flips jointly to input + target during I2I.")
    parser.add_argument("--vis_samples_per_epoch", type=int, default=0,
                        help="Number of I2I samples to visualise per epoch (0 to disable).")
    parser.add_argument("--vis_output_subdir", type=str, default="visualizations",
                        help="Subdirectory under the run output dir for visualisation images.")

    # Checkpointing & fault tolerance
    parser.add_argument("--smart_checkpoint", action="store_true", default=False,
                        help="Only keep 'latest' and 'best'; gracefully recover on save failure.")
    parser.add_argument("--checkpoint_errors_fatal", action="store_true", default=False,
                        help="Abort on checkpoint-save failure (default: keep training).")

    # Memory & speed knobs
    parser.add_argument("--enable_gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--enable_memory_efficient_attention", action="store_true", default=True)

    # Variable-resolution mode
    parser.add_argument("--variable_resolution", action="store_true", default=False,
                        help="Keep original resolutions (I2I only). Best paired with batch_size=1.")
    parser.add_argument("--post_bokeme_syn", action="store_true", default=False,
                        help="Only use the post_bokeme JSONL with online BokehMe synthesis (I2I-only mode).")
    parser.add_argument("--include_post_bokeme_syn", action="store_true", default=False,
                        help="Add the BokehMe-failure synthetic dataset to the regular I2I mix.")

    args = parser.parse_args()
    if isinstance(args.camera_anns, str):
        args.camera_anns = args.camera_anns.split(",") if args.camera_anns else []
    args.synthetic_sample_num = args.train_batch_size - 1
    args.perform_swap = args.swap_prob > 0.0

    if args.optimizer == "prodigy":
        args.learning_rate = 1.0
    elif args.optimizer == "lion":
        args.learning_rate = args.learning_rate * 0.1
        args.weight_decay = args.weight_decay * 10

    args.blocks = [FLUX_TRANSFORMER_BLOCKS[int(i)] for i in args.block_ids]
    return args


# ---------------------------------------------------------------------------
# Online BokehMe-synthesis dataset (uses all-in-focus + depth pairs)
# ---------------------------------------------------------------------------
class PostBokehFailureDataset(Dataset):
    """Dataset for on-the-fly BokehMe synthesis from all-in-focus + depth pairs."""

    def __init__(self, jsonl_path, camera_anns, size=512, horizontal_flip=True, tokenizer=None):
        self.jsonl_path = jsonl_path
        self.size = size
        self.horizontal_flip = horizontal_flip
        self.tokenizer = tokenizer
        if isinstance(camera_anns, str):
            camera_anns = camera_anns.split(",")
        self.camera_anns = list(camera_anns or [])

        self.samples = []
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Post BokehMe JSONL not found: {jsonl_path}")

        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                image_path = payload.get("all_in_focus_path") or payload.get("image_path")
                depth_path = payload.get("final_depth") or payload.get("depth_path")
                if not image_path or not depth_path:
                    continue
                if not (os.path.exists(image_path) and os.path.exists(depth_path)):
                    continue
                self.samples.append({
                    "image_path": image_path,
                    "depth_path": depth_path,
                    "meta": payload,
                })

        if len(self.samples) == 0:
            raise ValueError(f"No usable samples located in {jsonl_path}")

    @staticmethod
    def _get_resize_crop_params(width, height, target_size):
        if target_size is None:
            return {"ratio": 1.0, "new_w": width, "new_h": height, "left": 0, "top": 0}
        if width == target_size and height == target_size:
            return {"ratio": 1.0, "new_w": width, "new_h": height, "left": 0, "top": 0}
        ratio = target_size / max(1, min(width, height))
        new_w = int(np.ceil(width * ratio))
        new_h = int(np.ceil(height * ratio))
        left = max((new_w - target_size) // 2, 0)
        top = max((new_h - target_size) // 2, 0)
        left = min(left, max(new_w - target_size, 0))
        top = min(top, max(new_h - target_size, 0))
        return {"ratio": ratio, "new_w": new_w, "new_h": new_h, "left": left, "top": top}

    @staticmethod
    def _load_depth(depth_path, fallback_shape):
        depth = None
        try:
            if depth_path.endswith(".npz"):
                with np.load(depth_path) as data:
                    if "depth" in data:
                        depth = data["depth"]
                    elif len(data.files) > 0:
                        depth = data[data.files[0]]
            elif depth_path.endswith(".npy"):
                depth = np.load(depth_path)
            else:
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        except Exception:
            depth = None

        if depth is None:
            depth = np.zeros(fallback_shape, dtype=np.float32)

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        return depth

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        meta = sample["meta"]
        image_path = sample["image_path"]
        depth_path = sample["depth_path"]

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = image_rgb.shape[:2]

        params = self._get_resize_crop_params(original_w, original_h, self.size)
        if self.size is not None:
            resized = cv2.resize(image_rgb, (params["new_w"], params["new_h"]), interpolation=cv2.INTER_AREA)
            top, left = params["top"], params["left"]
            image_proc = resized[top:top + self.size, left:left + self.size]
        else:
            image_proc = image_rgb

        depth = self._load_depth(depth_path, (original_h, original_w))
        if self.size is not None:
            depth_resized = cv2.resize(depth, (params["new_w"], params["new_h"]), interpolation=cv2.INTER_LINEAR)
            top, left = params["top"], params["left"]
            depth_proc = depth_resized[top:top + self.size, left:left + self.size]
        else:
            depth_proc = depth
            if depth_proc.shape[:2] != image_proc.shape[:2]:
                depth_proc = cv2.resize(
                    depth_proc, (image_proc.shape[1], image_proc.shape[0]), interpolation=cv2.INTER_LINEAR
                )

        mask_proc = np.ones(image_proc.shape[:2], dtype=np.uint8) * 255

        flip_flag = self.horizontal_flip and (random.random() < 0.5)
        if flip_flag:
            image_proc = cv2.flip(image_proc, 1)
            depth_proc = cv2.flip(depth_proc, 1)
            mask_proc = cv2.flip(mask_proc, 1)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image_proc)).permute(2, 0, 1).float().div(255.0)
        image_tensor = image_tensor * 2.0 - 1.0

        fg_mask_tensor = torch.from_numpy(mask_proc).unsqueeze(0).float().div(255.0).round()
        depth_tensor = torch.from_numpy(np.ascontiguousarray(depth_proc)).unsqueeze(0).float()

        if self.tokenizer is not None:
            inputs = self.tokenizer(
                "", max_length=self.tokenizer.model_max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            text_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        else:
            text_ids = torch.zeros(1, 0)
            attention_mask = torch.zeros(1, 0)

        if self.size is not None:
            crop_params_norm = {
                "ratio": params["ratio"],
                "left_norm": params["left"] / max(1, params["new_w"]),
                "top_norm": params["top"] / max(1, params["new_h"]),
                "crop_width_norm": self.size / max(1, params["new_w"]),
                "crop_height_norm": self.size / max(1, params["new_h"]),
            }
        else:
            crop_params_norm = {
                "ratio": 1.0, "left_norm": 0.0, "top_norm": 0.0,
                "crop_width_norm": 1.0, "crop_height_norm": 1.0,
            }

        camera_ann = [0.5 for _ in self.camera_anns] if self.camera_anns else [0.5]
        dataset_type = meta.get("dataset", "post_bokeme_syn")

        return {
            "image": image_tensor,
            "image_np": np.ascontiguousarray(image_proc),
            "fg_mask": fg_mask_tensor,
            "fg_mask_np": np.ascontiguousarray(mask_proc),
            "depth_map": depth_tensor,
            "camera_ann": camera_ann,
            "caption": "",
            "text_input_ids": text_ids,
            "text_attention_mask": attention_mask,
            "target_image_path": None,
            "dataset_type": dataset_type,
            "foreground_clear": True,
            "flip_flag": flip_flag,
            "crop_params_norm": crop_params_norm,
        }


# ---------------------------------------------------------------------------
# Small helpers shared by the training loop
# ---------------------------------------------------------------------------
def get_next_batch(dataloader_iter, dataloader):
    """Wrap a dataloader so we transparently restart on StopIteration."""
    try:
        batch = next(dataloader_iter)
        return batch, dataloader_iter, False
    except StopIteration:
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)
        return batch, dataloader_iter, True


def pick_real_or_synth(accelerator, real_data_prob):
    """Sample a real-vs-synth decision on rank 0 and broadcast it to all ranks."""
    if accelerator.is_main_process:
        do_real_val = (random.random() < real_data_prob)
        do_real_tensor = torch.tensor(int(do_real_val), device=accelerator.device)
    else:
        do_real_tensor = torch.tensor(0, device=accelerator.device)
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(do_real_tensor, src=0)
    return bool(do_real_tensor.item())


def _encode_prompt_with_t5(text_encoder, tokenizer, prompt=None, num_images_per_prompt=1, device=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    text_input_ids = tokenizer(
        prompt, padding="max_length", max_length=512, truncation=True,
        return_length=False, return_overflowing_tokens=False, return_tensors="pt",
    ).input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(
        batch_size * num_images_per_prompt, seq_len, -1
    )
    return prompt_embeds


def _encode_prompt_with_clip(text_encoder, tokenizer, prompt, num_images_per_prompt=1, device=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    text_inputs_ids = tokenizer(
        prompt, padding="max_length", max_length=77, truncation=True,
        return_overflowing_tokens=False, return_length=False, return_tensors="pt",
    ).input_ids
    prompt_embeds = text_encoder(text_inputs_ids.to(device), output_hidden_states=False)
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(batch_size * num_images_per_prompt, -1)
    return prompt_embeds


def encode_prompt(text_encoders, tokenizers, prompt, device=None, num_images_per_prompt=1):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype
    device = device if device is not None else text_encoders[1].device
    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0], tokenizer=tokenizers[0],
        prompt=prompt, num_images_per_prompt=num_images_per_prompt, device=device,
    )
    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1], tokenizer=tokenizers[1],
        prompt=prompt, num_images_per_prompt=num_images_per_prompt, device=device,
    )
    text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def compute_text_embeddings(prompt, text_encoders, tokenizers, accelerator):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(text_encoders, tokenizers, prompt)
        prompt_embeds = prompt_embeds.to(accelerator.device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        text_ids = text_ids.to(accelerator.device)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a [-1, 1] tensor to a uint8 HxWx3 numpy array for visualisation."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.detach().float().cpu().clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.permute(1, 2, 0).numpy()
    return np.clip(tensor * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # I2I-only mode is forced by variable-resolution training and by the
    # post-BokehMe-synthesis-only run; in those cases we skip T2I entirely.
    i2i_only_mode = args.variable_resolution or args.post_bokeme_syn

    time_str = time.strftime("%Y%m%d_%H%M%S")
    exec_name = (
        f"{time_str}-flux-blocks:{compress_block_ids(args.block_ids)}"
        f"-lora_rank:{args.lora_rank}-anns:{'_'.join(args.camera_anns)}"
    )
    print(f"Executing {exec_name}")
    output_dir = os.path.join(args.output_dir, exec_name)
    accelerator_project_config = ProjectConfiguration(project_dir=output_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    vis_output_dir: Optional[Path] = None
    if args.vis_samples_per_epoch > 0 and args.vis_output_subdir:
        vis_output_dir = Path(output_dir) / args.vis_output_subdir
        vis_output_dir.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.post_bokeme_syn:
            jsonl_name = os.path.basename(args.post_bokeme_jsonl or "")
            print(f"Post-BokehMe synthesis mode: target images are produced online from {jsonl_name}.")
        print(f"Variable-resolution training: {'enabled' if args.variable_resolution else 'disabled'}")
        print(f"I2I horizontal flip: {'enabled' if args.i2i_horizontal_flip else 'disabled'}")
        if not i2i_only_mode:
            print(f"T2I grounded attention (swap): {'enabled' if args.perform_swap else 'disabled (default)'}")
        elif args.variable_resolution:
            print("Variable-resolution mode runs I2I only and keeps the original image resolution.")
            print("Variable-resolution mode uses a memory-fallback VAE encode (auto-downscales on OOM).")
            if args.train_batch_size > 1:
                print(
                    f"[hint] Variable-resolution mode is happiest with batch_size=1; current value is "
                    f"{args.train_batch_size}."
                )
        elif args.post_bokeme_syn:
            print("post_bokeme_syn skips all T2I batches and uses BokehMe online synthesis as I2I targets.")

    # ----- Precision-aware base-model load -----
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        print("Loading FluxKontextPipeline...")

    # main_process_first lets rank 0 populate the HF cache before peers read it.
    with accelerator.main_process_first():
        kontext_pipeline = FluxKontextPipeline.from_pretrained(
            args.pretrained_model_name_or_path, torch_dtype=weight_dtype
        )
    tokenizer_one = kontext_pipeline.tokenizer
    tokenizer_two = kontext_pipeline.tokenizer_2
    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoder_one = kontext_pipeline.text_encoder
    text_encoder_two = kontext_pipeline.text_encoder_2
    text_encoders = [text_encoder_one, text_encoder_two]
    noise_scheduler = kontext_pipeline.scheduler
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae = kontext_pipeline.vae
    transformer = kontext_pipeline.transformer
    if args.enable_gradient_checkpointing:
        try:
            transformer.enable_gradient_checkpointing()
        except Exception:
            pass
    if args.enable_memory_efficient_attention:
        try:
            if hasattr(transformer, "set_attention_slice"):
                transformer.set_attention_slice("max")
            elif hasattr(transformer, "enable_attention_slicing"):
                transformer.enable_attention_slicing("max")
        except Exception:
            pass

    # Freeze every base-model parameter; only adapter modules will train
    text_encoders[0].requires_grad_(False)
    text_encoders[1].requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    kontext_pipeline = kontext_pipeline.to(accelerator.device, dtype=weight_dtype)
    need_guidance = getattr(transformer.config, "guidance_embeds", False)
    vae_scale_factor = getattr(kontext_pipeline, "vae_scale_factor", getattr(vae, "scale_factor", 8))

    def _flux_floor_to_multiple(x: torch.Tensor, multiple: int | None = None) -> torch.Tensor:
        """Floor the spatial dims of an image batch to ``multiple`` (default 16).

        FLUX uses an 8x VAE plus a 2x2 packing step, so the VAE input must be
        divisible by 16 in both height and width.
        """
        if multiple is None:
            multiple = int(vae_scale_factor) * 2
        B, C, H, W = x.shape
        new_H = (H // multiple) * multiple
        new_W = (W // multiple) * multiple
        if (new_H != H) or (new_W != W):
            top = (H - new_H) // 2
            left = (W - new_W) // 2
            x = x[..., top:top + new_H, left:left + new_W]
        return x

    # Refresh module handles in case ``.to()`` returned wrapped objects
    text_encoders[0] = kontext_pipeline.text_encoder
    text_encoders[1] = kontext_pipeline.text_encoder_2
    vae = kontext_pipeline.vae
    transformer = kontext_pipeline.transformer

    # ----- Bokeh adapter -----
    bokeh_adapter = BokehFluxControlAdapter(
        transformer,
        blocks=args.blocks,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        unfreeze_q=args.unfreeze_q,
        unfreeze_k=args.unfreeze_k,
    )
    params_to_opt = itertools.chain(
        bokeh_adapter.embedding_layer.parameters(),
        bokeh_adapter.adapter_modules.parameters(),
    )
    if accelerator.is_main_process:
        print(bokeh_adapter)

    num_trainable_params = sum(p.numel() for p in bokeh_adapter.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        print(f"Number of trainable parameters: {num_trainable_params}")

    # ----- BokehMe synthesis components (used by the T2I synthetic path and by I2I fallback) -----
    classical_renderer = ModuleRenderScatter().to(accelerator.device)
    arnet = ARNet(2, 5, 4, 128, 3, False, "distinct_source", False, "elu").to(accelerator.device)
    iunet = IUNet(2, 8, 3, 64, 3, False, "distinct_source", False, "elu").to(accelerator.device)
    arnet.load_state_dict(torch.load(args.arnet_ckpt, weights_only=False, map_location="cpu")["model"])
    iunet.load_state_dict(torch.load(args.iunet_ckpt, weights_only=False, map_location="cpu")["model"])
    arnet.eval()
    iunet.eval()

    real_dataset = real_dataloader = None
    synth_dataset = synth_dataloader = None

    if not args.variable_resolution and not args.post_bokeme_syn:
        if not args.itw_jsonl:
            raise ValueError("--itw_jsonl is required for T2I training (omit it only with I2I-only modes).")
        # Unified ITW jsonl: real vs. synth distinguished by the suitable_for_synthetic flag
        real_base = FlickrInTheWildDataset(
            args.itw_jsonl, args.camera_anns,
            is_main_process=accelerator.is_main_process, size=args.size,
            synthetic_pairing=False, filter_recency=True,
        )
        real_indices = [idx for idx, p in enumerate(real_base.photos) if not p.get("suitable_for_synthetic", False)]
        real_dataset = Subset(real_base, real_indices)
        real_dataloader = DataLoader(
            real_dataset, batch_size=args.train_batch_size, shuffle=True,
            num_workers=args.dataloader_num_workers, collate_fn=collate_fn,
        )

        synth_base = FlickrInTheWildDataset(
            args.itw_jsonl, args.camera_anns,
            is_main_process=accelerator.is_main_process, size=args.size,
            synthetic_pairing=True, filter_recency=True,
        )
        synth_indices = [idx for idx, p in enumerate(synth_base.photos) if p.get("suitable_for_synthetic", False)]
        synth_dataset = Subset(synth_base, synth_indices)
        synth_actual_batch_size = args.train_batch_size // (1 + args.synthetic_sample_num)

        def custom_collate(batch):
            return collate_fn_synthetic_pairs(
                batch, synth_sample=args.synthetic_sample_num,
                classical_renderer=classical_renderer, arnet=arnet, iunet=iunet,
                device=accelerator.device, is_main_process=accelerator.is_main_process,
                K_min=args.K_min, K_max=args.K_max,
            )

        synth_dataloader = DataLoader(
            synth_dataset, batch_size=synth_actual_batch_size, shuffle=True,
            num_workers=args.dataloader_num_workers, collate_fn=custom_collate,
        )

    i2i_dataset = None
    i2i_dataloader = None
    i2i_size = None if args.variable_resolution else args.size

    if args.post_bokeme_syn:
        if not args.post_bokeme_jsonl:
            raise ValueError("--post_bokeme_jsonl is required when --post_bokeme_syn is set.")
        post_dataset = PostBokehFailureDataset(
            jsonl_path=args.post_bokeme_jsonl, camera_anns=args.camera_anns,
            size=i2i_size, horizontal_flip=args.i2i_horizontal_flip, tokenizer=None,
        )
        i2i_dataset = post_dataset
        i2i_dataloader = DataLoader(
            post_dataset, batch_size=args.train_batch_size, shuffle=True,
            num_workers=args.dataloader_num_workers, collate_fn=collate_fn_i2i,
        )
        if accelerator.is_main_process:
            print(f"[OK] post_bokeme_syn: loaded {len(post_dataset)} samples for online BokehMe synthesis.")
    else:
        i2i_pre_paths = list(args.i2i_jsonl or [])
        i2i_datasets = []
        for p in i2i_pre_paths:
            if not os.path.exists(p):
                if accelerator.is_main_process:
                    print(f"[WARN] I2I JSONL not found, skip: {p}")
                continue
            ds = FlickrInTheWildDataset(
                p, args.camera_anns,
                is_main_process=accelerator.is_main_process, size=i2i_size,
                synthetic_pairing=False, filter_recency=False,
                horizontal_flip=args.i2i_horizontal_flip,
            )
            i2i_datasets.append(ds)
            if accelerator.is_main_process:
                dataset_name = os.path.basename(os.path.dirname(p)) or os.path.basename(p)
                flip_status = "on" if args.i2i_horizontal_flip else "off"
                print(f"[OK] Loaded I2I dataset {dataset_name} (horizontal flip: {flip_status}).")

        if args.include_post_bokeme_syn:
            if args.post_bokeme_jsonl and os.path.exists(args.post_bokeme_jsonl):
                post_dataset = PostBokehFailureDataset(
                    jsonl_path=args.post_bokeme_jsonl, camera_anns=args.camera_anns,
                    size=i2i_size, horizontal_flip=args.i2i_horizontal_flip, tokenizer=None,
                )
                i2i_datasets.append(post_dataset)
                if accelerator.is_main_process:
                    print(f"[OK] Loaded BokehMe-failure dataset {os.path.basename(args.post_bokeme_jsonl)}.")
            elif accelerator.is_main_process:
                print(f"[WARN] post_bokeme JSONL not found: {args.post_bokeme_jsonl}")

        if args.itw_jsonl and os.path.exists(args.itw_jsonl):
            itw_i2i_base = FlickrInTheWildDataset(
                args.itw_jsonl, args.camera_anns,
                is_main_process=accelerator.is_main_process, size=i2i_size,
                synthetic_pairing=True, filter_recency=True,
                horizontal_flip=args.i2i_horizontal_flip,
            )
            itw_i2i_indices = [idx for idx, p in enumerate(itw_i2i_base.photos) if p.get("suitable_for_synthetic", False)]
            if itw_i2i_indices:
                i2i_datasets.append(Subset(itw_i2i_base, itw_i2i_indices))

        if len(i2i_datasets) == 0:
            i2i_dataloader = None
        else:
            i2i_dataset = i2i_datasets[0] if len(i2i_datasets) == 1 else ConcatDataset(i2i_datasets)
            i2i_dataloader = DataLoader(
                i2i_dataset, batch_size=args.train_batch_size, shuffle=True,
                num_workers=args.dataloader_num_workers, collate_fn=collate_fn_i2i,
            )

    # ----- Step accounting -----
    if i2i_only_mode:
        t2i_steps = 0
        i2i_steps = len(i2i_dataloader) if i2i_dataloader is not None else 0
        steps_per_epoch = i2i_steps // max(1, accelerator.num_processes)
        real_data_prob = 0.0
    else:
        t2i_steps = (
            max(len(real_dataloader), len(synth_dataloader))
            if (real_dataloader and synth_dataloader and len(real_dataloader) and len(synth_dataloader))
            else (len(real_dataloader) if real_dataloader else 0) or (len(synth_dataloader) if synth_dataloader else 0)
        )
        i2i_steps = len(i2i_dataloader) if i2i_dataloader is not None else 0
        steps_per_epoch = (t2i_steps + i2i_steps) // max(1, accelerator.num_processes)
        real_data_prob = args.real_data_ratio

    # ----- Optimiser & LR schedule -----
    if args.optimizer == "prodigy":
        optimizer = prodigyopt.Prodigy(
            params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay,
            use_bias_correction=True, safeguard_warmup=True,
        )
    elif args.optimizer == "lion":
        optimizer = Lion(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    if accelerator.is_main_process:
        print(f"Using {args.optimizer} optimizer (lr={args.learning_rate}, weight_decay={args.weight_decay}).")
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_epochs * steps_per_epoch,
    )

    if accelerator.is_main_process:
        if args.post_bokeme_syn:
            post_size = len(i2i_dataset) if i2i_dataset is not None else 0
            print(f"post_bokeme_syn: I2I-only, {post_size} samples (online BokehMe synthesis).")
        elif args.variable_resolution:
            print(f"Variable resolution: I2I-only, {len(i2i_dataset) if i2i_dataset is not None else 0} samples.")
        else:
            real_size = len(real_dataset) if real_dataset is not None else 0
            synth_size = len(synth_dataset) if synth_dataset is not None else 0
            print(f"Real dataset: {real_size} samples. Synthetic dataset: {synth_size} samples.")

    # Total dataset size logged to trackers
    if i2i_only_mode:
        total_dataset_size = len(i2i_dataset) if i2i_dataset is not None else 0
    else:
        real_size = len(real_dataset) if real_dataset is not None else 0
        synth_size = len(synth_dataset) if synth_dataset is not None else 0
        total_dataset_size = real_size + synth_size

    accelerator.init_trackers(
        project_name="flux-bokeh_adapter",
        config={
            **args.__dict__,
            "num_trainable_params": num_trainable_params,
            "dataset_size": total_dataset_size,
            "real_data_prob": real_data_prob,
        },
        init_kwargs=(
            {"wandb": {"id": args.wandb_resume_id, "resume": "allow"}}
            if args.wandb_resume_id is not None else {}
        ),
    )

    try:
        wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
    except (ValueError, KeyError):
        wandb_tracker = None

    # Prepare with accelerate; whether i2i_dataloader is included depends on mode
    if i2i_only_mode:
        if i2i_dataloader is not None:
            bokeh_adapter, optimizer, lr_scheduler, i2i_dataloader = accelerator.prepare(
                bokeh_adapter, optimizer, lr_scheduler, i2i_dataloader
            )
        else:
            bokeh_adapter, optimizer, lr_scheduler = accelerator.prepare(bokeh_adapter, optimizer, lr_scheduler)
        real_dataloader = None
        synth_dataloader = None
    else:
        if i2i_dataloader is not None:
            (bokeh_adapter, optimizer, lr_scheduler,
             real_dataloader, synth_dataloader, i2i_dataloader) = accelerator.prepare(
                bokeh_adapter, optimizer, lr_scheduler, real_dataloader, synth_dataloader, i2i_dataloader
            )
        else:
            (bokeh_adapter, optimizer, lr_scheduler,
             real_dataloader, synth_dataloader) = accelerator.prepare(
                bokeh_adapter, optimizer, lr_scheduler, real_dataloader, synth_dataloader
            )

    if args.load_checkpoint is not None:
        try:
            accelerator.load_state(args.load_checkpoint)
            if accelerator.is_main_process:
                print(f"[OK] Restored accelerator state from {args.load_checkpoint}.")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"[WARN] Failed to restore accelerator state; continuing fresh: {e}")

    # Resilient global_step recovery (handles 'latest' / 'best' / numeric dirs)
    global_step = 0
    if args.load_checkpoint is not None:
        try:
            ckpt_base = os.path.basename(os.path.normpath(args.load_checkpoint))
            if ckpt_base.isdigit():
                global_step = int(ckpt_base)
            else:
                parent_dir = os.path.dirname(os.path.normpath(args.load_checkpoint))
                step_loaded = False
                for info_name in ["latest_info.json", "best_info.json"]:
                    info_path = os.path.join(parent_dir, info_name)
                    if os.path.exists(info_path):
                        try:
                            with open(info_path, "r") as f:
                                info = json.load(f)
                            if "step" in info:
                                global_step = int(info["step"])
                                step_loaded = True
                                break
                        except Exception:
                            pass
                if not step_loaded:
                    global_step = 0
        except Exception:
            global_step = 0

    # Tracker used by smart_checkpoint to remember the best-loss snapshot
    best_loss_tracker = None
    if args.smart_checkpoint:
        best_loss_tracker = {"best_loss": None, "best_step": None}

    # ---------------- Smart checkpoint save with crash-safe rollback ----------------
    def save_checkpoint(accelerator, output_dir, step, avg_loss, best_tracker, is_final=False, errors_fatal=False):
        latest_path = os.path.join(output_dir, "latest")
        best_path = os.path.join(output_dir, "best")

        if accelerator.is_main_process and os.path.exists(latest_path):
            shutil.rmtree(latest_path)
        accelerator.wait_for_everyone()

        latest_saved_ok = True
        try:
            accelerator.save_state(latest_path, safe_serialization=False)
        except Exception as e:
            latest_saved_ok = False
            if accelerator.is_main_process:
                print(f"[WARN] Failed to save latest, falling back to a step-named dir: {e}")
            if not errors_fatal:
                try:
                    fallback_path = os.path.join(output_dir, f"{step}")
                    accelerator.save_state(fallback_path, safe_serialization=False)
                    if accelerator.is_main_process:
                        print(f"[OK] Fallback save succeeded at {fallback_path}.")
                except Exception as e2:
                    if accelerator.is_main_process:
                        print(f"[WARN] Fallback save also failed; training continues: {e2}")
            else:
                raise

        accelerator.wait_for_everyone()

        is_new_best = False
        if accelerator.is_main_process and latest_saved_ok:
            save_info = {"step": step, "loss": avg_loss, "timestamp": time.strftime("%Y%m%d_%H%M%S")}

            latest_info_path = os.path.join(output_dir, "latest_info.json")
            try:
                with open(latest_info_path, "w") as f:
                    json.dump(save_info, f, indent=2)
            except Exception as e:
                print(f"[WARN] Failed to write latest_info.json: {e}")
                if errors_fatal:
                    raise

            if best_tracker["best_loss"] is None or avg_loss < best_tracker["best_loss"]:
                best_tracker["best_loss"] = avg_loss
                best_tracker["best_step"] = step
                is_new_best = True

                temp_best_path = os.path.join(output_dir, f".best_tmp_{step}")
                backup_best_path = os.path.join(output_dir, f".best_backup_{step}")
                if os.path.exists(temp_best_path):
                    shutil.rmtree(temp_best_path)
                if os.path.exists(backup_best_path):
                    shutil.rmtree(backup_best_path)

                best_updated_successfully = False
                try:
                    shutil.copytree(latest_path, temp_best_path)
                    if os.path.exists(best_path):
                        os.replace(best_path, backup_best_path)
                    os.replace(temp_best_path, best_path)
                    if os.path.exists(backup_best_path):
                        shutil.rmtree(backup_best_path)
                    best_updated_successfully = True
                except Exception as e:
                    # Roll back to the previous 'best' if the swap failed mid-way
                    try:
                        if (not os.path.exists(best_path)) and os.path.exists(backup_best_path):
                            os.replace(backup_best_path, best_path)
                    except Exception:
                        pass
                    if os.path.exists(temp_best_path):
                        shutil.rmtree(temp_best_path)

                    candidates_dir = os.path.join(output_dir, "best_candidates")
                    os.makedirs(candidates_dir, exist_ok=True)
                    fallback_path = os.path.join(candidates_dir, f"step_{step}")
                    if os.path.exists(fallback_path):
                        shutil.rmtree(fallback_path)
                    try:
                        shutil.copytree(latest_path, fallback_path)
                        print(
                            f"[WARN] Best update failed; the new candidate was preserved at {fallback_path}. "
                            f"Reason: {e}"
                        )
                    except Exception as e_fallback:
                        print(f"[WARN] Best update failed and candidate save failed: {e} / {e_fallback}")

                    best_updated_successfully = False

                if best_updated_successfully:
                    best_info_path = os.path.join(output_dir, "best_info.json")
                    try:
                        with open(best_info_path, "w") as f:
                            json.dump(save_info, f, indent=2)
                    except Exception as e:
                        print(f"[WARN] Failed to write best_info.json: {e}")
                        if errors_fatal:
                            raise
                    print(f"[BEST] New best model! Loss: {avg_loss:.6f} -> saved to {best_path}")
                else:
                    print("[INFO] This step beats history but 'best' could not be replaced. See best_candidates/.")

            status = "NEW BEST" if is_new_best else "LATEST"
            print(f"[{status}] Checkpoint saved:")
            print(f"  step: {step}, loss: {avg_loss:.6f}")
            print(f"  latest: {latest_path}")
            if best_tracker["best_loss"] is not None:
                print(
                    f"  best:   {best_path} (loss: {best_tracker['best_loss']:.6f} @ step "
                    f"{best_tracker['best_step']})"
                )

            if is_final:
                print("[DONE] Training finished. Final state:")
                print(f"  - latest model: {latest_path}")
                if best_tracker["best_loss"] is not None:
                    print(f"  - best model:   {best_path} (loss: {best_tracker['best_loss']:.6f})")

        accelerator.wait_for_everyone()

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # ---------------- Memory-fallback wrappers (variable-resolution mode) ----------------
    def smart_resize_for_memory(imgs, target_short_side=512):
        """Resize so the shorter side equals ``target_short_side``, preserving aspect ratio.

        Returns the original tensor untouched if it is already small enough.
        """
        with torch.no_grad():
            B, C, H, W = imgs.shape
            short_side = min(H, W)
            if short_side <= target_short_side:
                return imgs

            scale = target_short_side / short_side
            new_H = max(16, (int(H * scale) // 16) * 16)
            new_W = max(16, (int(W * scale) // 16) * 16)
            return F.interpolate(imgs, size=(new_H, new_W), mode="bilinear", align_corners=False)

    def safe_vae_encode_with_fallback(vae, imgs, accelerator, max_retries=6):
        """VAE encode with progressive down-scaling on CUDA OOM."""
        original_size = imgs.shape
        current_imgs = imgs

        for attempt in range(max_retries):
            try:
                with torch.no_grad():
                    latents = vae.encode(current_imgs).latent_dist.sample()
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

                if attempt > 0 and accelerator.is_main_process:
                    print(
                        f"[WARN] VAE encode succeeded after downscaling from {original_size[2:]} to "
                        f"{current_imgs.shape[2:]}."
                    )
                return latents, current_imgs.shape[2:]

            except torch.cuda.OutOfMemoryError as e:
                if attempt == max_retries - 1:
                    if accelerator.is_main_process:
                        print(
                            f"[ERROR] VAE encode failed even after downscaling. Skipping batch. "
                            f"Original: {original_size[2:]}, last attempt: {current_imgs.shape[2:]}; {e}"
                        )
                    return None, None

                if "latents" in locals():
                    del latents
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                current_short_side = min(current_imgs.shape[2], current_imgs.shape[3])
                if attempt == 0:
                    new_target = max(128, int(current_short_side * 0.5))
                elif attempt == 1:
                    new_target = 1024
                elif attempt == 2:
                    new_target = 512
                elif attempt == 3:
                    new_target = 256
                else:
                    new_target = 128

                if accelerator.is_main_process:
                    print(
                        f"[WARN] VAE encode OOM (attempt {attempt + 1}/{max_retries}); "
                        f"shrinking shorter side {current_short_side} -> {new_target}."
                    )

                with torch.no_grad():
                    current_imgs = smart_resize_for_memory(current_imgs, new_target)
                    current_imgs = _flux_floor_to_multiple(current_imgs)

        return None, None  # unreachable

    def safe_transformer_forward_with_fallback(bokeh_adapter, transformer, latents, accelerator, max_retries=2, **kwargs):
        """Adapter forward pass with simple OOM-retry."""
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                model_pred = bokeh_adapter(transformer, hidden_states=latents, **kwargs)
                if attempt > 0 and accelerator.is_main_process:
                    print(f"[WARN] Transformer fallback succeeded on attempt {attempt + 1}.")
                return model_pred
            except torch.cuda.OutOfMemoryError as e:
                if attempt == max_retries - 1:
                    if accelerator.is_main_process:
                        print(f"[ERROR] Transformer forward failed after {max_retries} retries; skipping batch. {e}")
                    return None
                if "model_pred" in locals():
                    del model_pred
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                if accelerator.is_main_process:
                    print(f"[WARN] Transformer OOM (attempt {attempt + 1}/{max_retries}); retrying after cache flush.")
        return None  # unreachable

    # ---------------- Train! ----------------
    if args.variable_resolution:
        real_iter = synth_iter = None
        i2i_iter = iter(i2i_dataloader) if i2i_dataloader is not None else None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    else:
        real_iter = iter(real_dataloader) if real_dataloader is not None else None
        synth_iter = iter(synth_dataloader) if synth_dataloader is not None else None
        i2i_iter = iter(i2i_dataloader) if i2i_dataloader is not None else None

    if accelerator.is_main_process:
        print(f"Training on device: {accelerator.device}, mixed precision: {accelerator.mixed_precision}")

    epoch_losses = [] if args.smart_checkpoint else None
    avg_loss = float("nan")

    for epoch in range(global_step // max(1, steps_per_epoch), args.max_train_epochs):
        if accelerator.is_main_process:
            begin = time.perf_counter()
            itw_times = []
            synth_times = []
        vis_samples_logged = 0

        # Build the per-epoch I2I / T2I schedule and broadcast it from rank 0
        if i2i_only_mode:
            t2i_target = 0
            i2i_target = steps_per_epoch
            schedule_tensor = torch.ones(steps_per_epoch, device=accelerator.device, dtype=torch.int32)
        else:
            t2i_target = int(steps_per_epoch * max(0.0, min(1.0, 1.0 - args.i2i_ratio)))
            i2i_target = steps_per_epoch - t2i_target
            if accelerator.is_main_process:
                schedule_list = [1] * i2i_target + [0] * t2i_target
                random.shuffle(schedule_list)
                schedule_tensor = torch.tensor(schedule_list, device=accelerator.device, dtype=torch.int32)
            else:
                schedule_tensor = torch.empty(steps_per_epoch, device=accelerator.device, dtype=torch.int32)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(schedule_tensor, src=0)

        for step in range(steps_per_epoch):
            # Decide T2I vs. I2I from the shared schedule
            use_i2i = bool(int(schedule_tensor[step].item())) if i2i_dataloader is not None else False

            if use_i2i:
                batch, i2i_iter, _ = get_next_batch(i2i_iter, i2i_dataloader)
                batch_mode = "I2I"
            else:
                do_real = pick_real_or_synth(accelerator, real_data_prob)
                if do_real:
                    batch, real_iter, _ = get_next_batch(real_iter, real_dataloader)
                else:
                    batch, synth_iter, _ = get_next_batch(synth_iter, synth_dataloader)
                batch_mode = "T2I"

            if accelerator.is_main_process:
                if batch_mode == "I2I":
                    print("processing i2i batch...")
                else:
                    print("processing synthetic batch..." if batch.get("is_synthetic", False) else "processing in-the-wild batch...")
                load_data_time = time.perf_counter() - begin
                batch_start_time = time.perf_counter()

            with accelerator.accumulate(bokeh_adapter):
                # If any rank's I2I batch is empty (all samples filtered out), every rank must skip together
                local_skip = 1 if (batch_mode == "I2I" and batch.get("skip_batch", False)) else 0
                skip_tensor = torch.tensor(local_skip, device=accelerator.device)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(skip_tensor, op=dist.ReduceOp.SUM)
                if skip_tensor.item() > 0:
                    continue

                # Source image -> VAE latents
                imgs = batch["images"].to(accelerator.device, dtype=weight_dtype)
                imgs = _flux_floor_to_multiple(imgs)

                if args.variable_resolution:
                    # Drop pathologically large frames so they don't OOM rank 0
                    B, C, H, W = imgs.shape
                    max_pixels = 4 * 1024 * 1024
                    local_skip_resolution = 1 if (H * W) > max_pixels else 0
                    skip_resolution_tensor = torch.tensor(local_skip_resolution, device=accelerator.device)
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(skip_resolution_tensor, op=dist.ReduceOp.SUM)
                    if skip_resolution_tensor.item() > 0:
                        if accelerator.is_main_process:
                            print(
                                f"[skip] variable-resolution: dropped batch with >{max_pixels:,} pixels."
                            )
                        continue

                if args.variable_resolution:
                    if accelerator.is_main_process:
                        print(f"[vr] processing image size: {imgs.shape[2:]}")
                    latents, actual_img_size = safe_vae_encode_with_fallback(vae, imgs, accelerator)
                    if latents is None:
                        if accelerator.is_main_process:
                            print("[skip] variable-resolution: VAE encode failed.")
                        continue
                else:
                    with torch.no_grad():
                        latents = vae.encode(imgs).latent_dist.sample()
                        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                    actual_img_size = imgs.shape[2:]

                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

                # Noise & timesteps
                bsz = latents.shape[0]
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # Offset noise: https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                if batch.get("is_synthetic", False):
                    noise = noise[batch["batch_swap_ids"]]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme, batch_size=bsz,
                    logit_mean=0.0, logit_std=1.0, mode_scale=1.29,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=latents.device)

                if batch.get("is_synthetic", False):
                    timesteps = timesteps[batch["batch_swap_ids"]]

                # Flow-matching noised latents: z_t = (1 - sigma) * x + sigma * z1
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                # Pack noisy latents + image ids (Kontext helpers)
                B, C, H, W = noisy_latents.shape
                latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    B, H // 2, W // 2, accelerator.device, transformer.dtype
                )
                packed_noisy_latents = FluxKontextPipeline._pack_latents(noisy_latents, B, C, H, W)
                target_token_len = packed_noisy_latents.shape[1]

                assert packed_noisy_latents.shape[1] == latent_image_ids.shape[0], (
                    f"Target packing mismatch: packed_tokens={packed_noisy_latents.shape[1]}, "
                    f"img_ids={latent_image_ids.shape[0]}"
                )

                # Guidance (only used by FLUX variants with guidance embeddings)
                guidance = (
                    torch.full((latents.shape[0],), args.guidance_scale,
                               device=accelerator.device, dtype=transformer.dtype)
                    if need_guidance else None
                )

                # Text embeddings: for synthetic batches we share one caption across the group
                if batch["is_synthetic"]:
                    first_caption = batch["captions"][0:1]
                    prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                        first_caption, text_encoders, tokenizers, accelerator
                    )
                    batch_size = len(batch["captions"])
                    prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
                    pooled_prompt_embeds = pooled_prompt_embeds.expand(batch_size, -1)
                    prompt_embeds = prompt_embeds.to(dtype=transformer.dtype, device=transformer.device)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=transformer.dtype, device=transformer.device)
                    text_ids = text_ids.to(dtype=transformer.dtype, device=transformer.device)
                else:
                    prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                        batch["captions"], text_encoders, tokenizers, accelerator
                    )
                    prompt_embeds = prompt_embeds.to(dtype=transformer.dtype, device=transformer.device)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=transformer.dtype, device=transformer.device)
                    text_ids = text_ids.to(dtype=transformer.dtype, device=transformer.device)

                # ---------------- I2I-specific path: re-pack target + reference latents ----------------
                if batch_mode == "I2I":
                    bsz_i2i = batch["images"].shape[0]
                    target_list = []
                    cam_ann_list = []

                    # Online BokehMe synthesis for the target image (uses the I2I input + depth)
                    for i in range(bsz_i2i):
                        K = random.uniform(args.K_min, args.K_max)
                        bokeh_t, _, _, _ = add_bokeh(
                            image_np=batch["image_np_list"][i],
                            fg_mask_np=batch["fg_mask_np_list"][i],
                            depth_map=batch["depth_maps"][i, 0].cpu().numpy(),
                            classical_renderer=classical_renderer,
                            arnet=arnet, iunet=iunet, device=accelerator.device,
                            K_min=K, K_max=K, is_main_process=accelerator.is_main_process,
                        )
                        target_list.append(bokeh_t)
                        cam_ann_list.append(torch.tensor([K / args.K_max], device=accelerator.device, dtype=weight_dtype))

                    target_imgs = torch.stack(target_list).to(accelerator.device, dtype=weight_dtype)
                    target_imgs = _flux_floor_to_multiple(target_imgs)

                    # Encode the (now-synthetic) target image as the diffusion target
                    if args.variable_resolution:
                        latents, actual_target_size = safe_vae_encode_with_fallback(vae, target_imgs, accelerator)
                        if latents is None:
                            continue
                    else:
                        with torch.no_grad():
                            latents = vae.encode(target_imgs).latent_dist.sample()
                            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                        actual_target_size = target_imgs.shape[2:]

                    sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                    B, C, H, W = noisy_latents.shape
                    latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                        B, H // 2, W // 2, accelerator.device, transformer.dtype
                    )
                    packed_noisy_latents = FluxKontextPipeline._pack_latents(noisy_latents, B, C, H, W)
                    target_token_len = packed_noisy_latents.shape[1]

                    assert packed_noisy_latents.shape[1] == latent_image_ids.shape[0], (
                        f"Target packing mismatch: packed_tokens={packed_noisy_latents.shape[1]}, "
                        f"img_ids={latent_image_ids.shape[0]}"
                    )

                    # Reference image (input) encoded as Kontext context (type=1 image ids)
                    ref_imgs = batch["images"].to(accelerator.device, dtype=weight_dtype)
                    ref_imgs = _flux_floor_to_multiple(ref_imgs)
                    if args.variable_resolution:
                        ref_latents, actual_ref_size = safe_vae_encode_with_fallback(vae, ref_imgs, accelerator)
                        if ref_latents is None:
                            continue
                    else:
                        with torch.no_grad():
                            ref_latents = vae.encode(ref_imgs).latent_dist.sample()
                            ref_latents = (ref_latents - vae.config.shift_factor) * vae.config.scaling_factor
                        actual_ref_size = ref_imgs.shape[2:]
                    Br, Cr, Hr, Wr = ref_latents.shape
                    image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                        Br, Hr // 2, Wr // 2, accelerator.device, transformer.dtype
                    )
                    try:
                        image_ids[..., 0] = 1  # type=1 marks reference-image tokens
                    except Exception:
                        pass

                    ref_packed = FluxKontextPipeline._pack_latents(ref_latents, Br, Cr, Hr, Wr)
                    assert ref_packed.shape[1] == image_ids.shape[0], (
                        f"Reference packing mismatch: packed_tokens={ref_packed.shape[1]}, "
                        f"img_ids={image_ids.shape[0]}"
                    )

                    # Concatenate target + reference along the sequence dim (target-first, matching the official pipeline)
                    full_hidden_states = torch.cat([packed_noisy_latents, ref_packed], dim=1)
                    full_img_ids = torch.cat([latent_image_ids, image_ids], dim=0)
                    assert full_hidden_states.shape[1] == full_img_ids.shape[0], (
                        f"Concatenation mismatch: hidden_states={full_hidden_states.shape[1]}, "
                        f"img_ids={full_img_ids.shape[0]}"
                    )
                    packed_noisy_latents = full_hidden_states
                    latent_image_ids = full_img_ids

                    camera_anns = torch.stack(cam_ann_list, dim=0).to(accelerator.device, dtype=weight_dtype)
                    if camera_anns.ndim == 1:
                        camera_anns = camera_anns.unsqueeze(-1)

                    # Re-render captions: keep the original one with probability ``i2i_prompt_keep_ratio``,
                    # otherwise plug the K value into a small set of templates.
                    keep_ratio = getattr(args, "i2i_prompt_keep_ratio", 0.35)
                    k_norm_tensor = camera_anns.detach().float().squeeze(-1).clamp(min=0.0)
                    if k_norm_tensor.ndim == 0:
                        k_norm_tensor = k_norm_tensor.unsqueeze(0)
                    K_values = (k_norm_tensor * args.K_max).cpu().tolist()
                    templates = [
                        "Set dof_cond = {value} (stronger background defocus), no changes to composition, lighting, or colors.",
                        "Increase dof_cond to {value} to enhance background blur; preserve subject sharpness and keep composition, lighting, colors unchanged.",
                        "Apply dof_cond = {value} for stronger bokeh; keep the main subject crisp, do not alter composition, lighting, or color tone.",
                        "Use dof_cond = {value} to intensify background defocus; maintain original framing, lighting, and color fidelity.",
                    ]
                    new_captions = []
                    for i, old_cap in enumerate(batch["captions"]):
                        if (old_cap is not None and str(old_cap).strip() != "") and (random.random() < keep_ratio):
                            new_captions.append(old_cap)
                        else:
                            tpl = random.choice(templates)
                            new_captions.append(tpl.format(value=f"{K_values[i]:.2f}"))
                    prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                        new_captions, text_encoders, tokenizers, accelerator
                    )
                else:
                    camera_anns = batch["camera_anns"].unsqueeze(-1).to(accelerator.device, dtype=weight_dtype)

                # Grounded-attention swap only fires on T2I synthetic batches
                perform_swap = (batch_mode == "T2I" and batch.get("is_synthetic", False) and args.perform_swap)

                # Adapter forward
                if args.variable_resolution:
                    model_pred = safe_transformer_forward_with_fallback(
                        bokeh_adapter, transformer, packed_noisy_latents, accelerator,
                        camera_ann=camera_anns,
                        perform_swap=perform_swap,
                        batch_swap_ids=batch.get("batch_swap_ids", None) if perform_swap else None,
                        timestep=timesteps / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids[0],
                        img_ids=latent_image_ids,
                        is_i2i=(batch_mode == "I2I"),
                    )
                    if model_pred is None:
                        continue
                else:
                    model_pred = bokeh_adapter(
                        transformer,
                        camera_ann=camera_anns,
                        perform_swap=perform_swap,
                        batch_swap_ids=batch.get("batch_swap_ids", None) if perform_swap else None,
                        hidden_states=packed_noisy_latents,
                        timestep=timesteps / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids[0],
                        img_ids=latent_image_ids,
                        is_i2i=(batch_mode == "I2I"),
                    )

                # Drop the reference-image tokens; only target tokens contribute to the loss
                assert model_pred.shape[1] >= target_token_len, "model_pred sequence is too short"
                model_pred = model_pred[:, : target_token_len]
                model_pred = FluxKontextPipeline._unpack_latents(
                    model_pred,
                    height=latents.shape[2] * vae_scale_factor,
                    width=latents.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = noise - latents  # flow-matching velocity target

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                # Optional I2I sample visualisation (wandb + on-disk grid)
                if (
                    args.vis_samples_per_epoch > 0
                    and batch_mode == "I2I"
                    and accelerator.is_main_process
                    and vis_samples_logged < args.vis_samples_per_epoch
                ):
                    remaining = args.vis_samples_per_epoch - vis_samples_logged
                    num_samples = min(target_imgs.shape[0], remaining)
                    scale = float(getattr(vae.config, "scaling_factor", 1.0))
                    shift = float(getattr(vae.config, "shift_factor", 0.0))
                    for local_idx in range(num_samples):
                        with torch.no_grad():
                            pred_latent = noise[local_idx : local_idx + 1] - model_pred[local_idx : local_idx + 1]
                            pred_latent = (pred_latent / scale) + shift
                            pred_latent = pred_latent.to(dtype=vae.dtype)
                            pred_image = vae.decode(pred_latent, return_dict=False)[0]

                        input_np = tensor_to_uint8_image(ref_imgs[local_idx : local_idx + 1])
                        target_np = tensor_to_uint8_image(target_imgs[local_idx : local_idx + 1])
                        pred_np = tensor_to_uint8_image(pred_image)
                        grid = np.concatenate([input_np, target_np, pred_np], axis=1)

                        if vis_output_dir is not None:
                            vis_path = vis_output_dir / f"epoch{epoch:04d}_sample{vis_samples_logged:04d}.png"
                            Image.fromarray(grid).save(vis_path)

                        if wandb_tracker is not None and wandb is not None:
                            caption = f"epoch {epoch} sample {vis_samples_logged}: input | target | prediction"
                            wandb_tracker.log(
                                {
                                    f"vis/epoch_{epoch:04d}_sample_{vis_samples_logged:03d}": wandb.Image(
                                        grid, caption=caption
                                    )
                                },
                                step=global_step,
                            )

                        vis_samples_logged += 1
                        if vis_samples_logged >= args.vis_samples_per_epoch:
                            break

                # Backprop with a try/except so a single failure doesn't kill the run
                try:
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_to_opt, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Safe loss gathering: fall back to local value if NCCL hiccups
                    try:
                        avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                    except Exception as e:
                        if accelerator.is_main_process:
                            print(f"[WARN] loss gather failed; using local loss only: {e}")
                        avg_loss = loss.item()
                except Exception as e:
                    if accelerator.is_main_process:
                        print(f"[WARN] backward step failed; skipping batch: {e}")
                    optimizer.zero_grad()
                    continue

                if args.smart_checkpoint and epoch_losses is not None:
                    epoch_losses.append(avg_loss)
                if accelerator.is_main_process:
                    batch_time = time.perf_counter() - batch_start_time
                    if batch.get("is_synthetic", False):
                        synth_times.append(batch_time)
                    else:
                        itw_times.append(batch_time)

                    accelerator.log(
                        {
                            "epoch": epoch, "step": step, "data_time": load_data_time,
                            "time": time.perf_counter() - begin, "step_loss": avg_loss,
                        },
                        step=global_step,
                    )
                    print(f"Epoch {epoch}, Step {step}, Loss: {avg_loss}")

            global_step += 1
            if accelerator.is_main_process:
                begin = time.perf_counter()

        if accelerator.is_main_process:
            print(f"Epoch {epoch} finished.")
            avg_itw_time = sum(itw_times) / len(itw_times) if itw_times else 0
            avg_synth_time = sum(synth_times) / len(synth_times) if synth_times else 0
            print("Timing stats:")
            print(f"  Average in-the-wild batch processing time: {avg_itw_time:.4f}s ({len(itw_times)} batches)")
            print(f"  Average synthetic batch processing time:   {avg_synth_time:.4f}s ({len(synth_times)} batches)")

            accelerator.log(
                {
                    "avg_itw_batch_time": avg_itw_time,
                    "avg_synth_batch_time": avg_synth_time,
                    "itw_batch_count": len(itw_times),
                    "synth_batch_count": len(synth_times),
                },
                step=global_step,
            )

        if (epoch + 1) % args.save_every_n_epochs == 0:
            if args.smart_checkpoint:
                epoch_avg_loss = (
                    sum(epoch_losses) / len(epoch_losses) if (epoch_losses and len(epoch_losses) > 0) else avg_loss
                )
                save_checkpoint(
                    accelerator, output_dir, global_step, epoch_avg_loss,
                    best_loss_tracker, errors_fatal=args.checkpoint_errors_fatal,
                )
                epoch_losses = []
            else:
                if accelerator.is_main_process:
                    print("Saving checkpoint...")
                save_path = os.path.join(output_dir, f"{global_step}")
                accelerator.save_state(save_path, safe_serialization=False)

    # ----- Final save -----
    if args.smart_checkpoint:
        final_avg_loss = (
            sum(epoch_losses) / len(epoch_losses) if (epoch_losses and len(epoch_losses) > 0) else avg_loss
        )
        save_checkpoint(
            accelerator, output_dir, global_step, final_avg_loss,
            best_loss_tracker, is_final=True, errors_fatal=args.checkpoint_errors_fatal,
        )
    else:
        save_path = os.path.join(output_dir, "last")
        accelerator.save_state(save_path, safe_serialization=False)
    accelerator.end_training()


if __name__ == "__main__":
    main()
