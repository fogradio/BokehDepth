"""
gen_bokeh_stack.py
==================

Batch-generate a calibrated bokeh defocus stack from each sample of a dataset,
using the Stage-1 bokeh-diffusion model (FLUX.1-Kontext base + the
``BokehFluxControlAdapter`` LoRA) in image-to-image mode. The same script
handles many supported datasets -- pick one via ``--dataset``.

Inputs
------
- ``--adapter_ckpt PATH``   Stage-1 LoRA adapter checkpoint (.bin / .safetensors
                            file, or a directory containing one of them).
- ``--pretrained_model_name_or_path NAME_OR_PATH``
                            FLUX Kontext base model (HuggingFace id or local path).
                            Default: ``black-forest-labs/FLUX.1-Kontext-dev``.
- ``--dataset NAME``        One of: kitti, nyuv2, eth3d, hypersim, etc
- ``--<dataset>_root PATH`` Root directory of the chosen dataset; each dataset
                            has its own dedicated flag (e.g.  ``--nyuv2_root`` for
                            nyuv2, ``--hypersim_root`` for hypersim, ...).
- ``--k_min`` / ``--k_max`` / ``--k_step``
                            Range and step of the bokeh-strength K values used
                            to build the stack. Default: 1.0 -> 30.0 step 5.0,
                            with the upper bound always included.
- ``--output_root PATH``    Where outputs are written; per-dataset defaults are
                            applied when this flag is omitted.

Outputs (per sample, under ``--output_root``)
---------------------------------------------
For each sample identified by its relative path in the source dataset, the
script creates:

    <output_root>/<sample_rel_path>/defocus_stack/0.png      # K = k_vals[0]
    <output_root>/<sample_rel_path>/defocus_stack/1.png      # K = k_vals[1]
    ...
    <output_root>/<sample_rel_path>/defocus_stack/stack_index.json

``stack_index.json`` records the per-frame K values together with metadata
(source RGB path, source depth path, prompt template, guidance / steps, ...).

At the end of a run a JSONL manifest is also written under ``--output_root``
(``manifest_bokeh_diffusion_<dataset>.jsonl`` for most datasets, or one
manifest per split for NYUv2). The manifest is consumed by the Stage-2
training / evaluation code.

Examples
--------
Single-process run on Virtual KITTI 2 with the default K sequence::

    python gen_bokeh_stack.py \\
        --dataset vkitti2 \\
        --adapter_ckpt ../weights/bokeh_lora.bin \\
        --rgb_root   /data/Virtual_KITTI_2/rgb \\
        --depth_root /data/Virtual_KITTI_2/depth \\
        --output_root /data/Virtual_KITTI_2/bokeh_diffusion_defocus_stack


Multi-GPU launch via ``accelerate`` (the script splits the dataset across ranks
with stride slicing for load balance)::

    accelerate launch --num_processes 4 gen_bokeh_stack.py \\
        --dataset hypersim \\
        --adapter_ckpt ../weights/bokeh_lora.bin \\
        --hypersim_root /data/ml-hypersim/hypersim_minimal \\
        --hypersim_split_csv /data/ml-hypersim/.../metadata_images_split_scene_v1.csv \\
        --hypersim_split all
"""

import os
import json
import argparse
import csv
from collections import Counter
from functools import lru_cache
from typing import List, Tuple, Optional, Dict, Any, Set

import numpy as np
import torch
from PIL import Image
import cv2
import h5py
from accelerate import Accelerator

# Fix: enable PIL's tolerant mode so we can load truncated / damaged image files.
# Useful when occasional corrupt files appear in a large-scale dataset.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Import bokeh-diffusion components (originally from eval_i2i_syn_foreground.py).
from constants import FLUX_TRANSFORMER_BLOCKS
from utils import parse_block_ids
from model.bokeh_adapter_flux import BokehFluxControlAdapter
from diffusers import FluxKontextPipeline


def parse_kitti_test_files(test_files_path: str) -> List[Tuple[str, int, str]]:
    items: List[Tuple[str, int, str]] = []
    if not os.path.exists(test_files_path):
        raise FileNotFoundError(f"KITTI test file list not found: {test_files_path}")
    with open(test_files_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            sequence_path, frame_id_str, camera_side = parts
            try:
                frame_id = int(frame_id_str)
            except ValueError:
                continue
            items.append((sequence_path, frame_id, camera_side))
    return items


def find_kitti_rgb_image(kitti_data_root: str, sequence_path: str, frame_id: int, camera_side: str) -> Optional[str]:
    try:
        date_str, sequence_name = sequence_path.split('/')
    except ValueError:
        return None
    camera_dir = 'image_02' if camera_side == 'l' else 'image_03'
    rgb_path = os.path.join(
        kitti_data_root,
        date_str,
        sequence_name,
        camera_dir,
        'data',
        f'{frame_id:010d}.png'
    )
    return rgb_path if os.path.exists(rgb_path) else None


def get_kitti_items_from_test_files(
    kitti_data_root: str,
    test_files_path: str,
) -> List[Tuple[str, int, str, str, int]]:
    test_items = parse_kitti_test_files(test_files_path)
    kitti_items: List[Tuple[str, int, str, str, int]] = []
    for index, (sequence_path, frame_id, camera_side) in enumerate(test_items):
        rgb_path = find_kitti_rgb_image(kitti_data_root, sequence_path, frame_id, camera_side)
        if rgb_path is None:
            print(f"[WARN] KITTI image not found: {sequence_path} frame {frame_id} side {camera_side}")
            continue
        kitti_items.append((sequence_path, frame_id, camera_side, rgb_path, index))
    return kitti_items


def load_kitti_gt_depths(npz_path: str) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    if 'data' in data:
        return data['data']
    # fallback: take first key
    first_key = list(data.keys())[0]
    return data[first_key]


def load_depth_from_npz_index(gt_depths_array: np.ndarray, index: int) -> np.ndarray:
    if index >= len(gt_depths_array):
        raise IndexError(f'gt_depths index out of bounds: {index} >= {len(gt_depths_array)}')
    depth = gt_depths_array[index]
    return depth.astype(np.float32)


def aperture_from_k_value(K_abs: float) -> float:
    """Convert a K value into an f-number (f/X.X)."""
    return max(1.0, 30.0 / K_abs)


def build_prompt_from_template(template: str, K_abs: float) -> str:
    """Build the final prompt from a template (add-bokeh task only).

    Placeholders supported: {aperture} (f-number), {K_abs} (absolute K),
    {value} (alias of absolute K).
    """
    aperture = aperture_from_k_value(K_abs)
    try:
        return template.format(aperture=aperture, K_abs=K_abs, value=K_abs)
    except Exception:
        # If the template is malformed, return it untouched.
        return template


def parse_args():
    parser = argparse.ArgumentParser(description="Generate bokeh defocus stacks using bokeh-diffusion in I2I mode")

    # bokeh-diffusion model configuration
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--block_ids", type=parse_block_ids, default="0-56")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    # Adapter checkpoint
    parser.add_argument("--adapter_ckpt", type=str, required=True,
                        help="Adapter checkpoint path (.bin file or a directory containing pytorch_model.bin)")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--unfreeze_q", action="store_true", default=True)
    parser.add_argument("--unfreeze_k", action="store_true", default=True)

    parser.add_argument('--dataset', type=str, choices=['vkitti2', 'kitti_test', 'kitti_train', 'nyuv2', 'eth3d', 'hypersim', 'make3d_kaggle', 'make3d_official', 'opensun3d', 'middlebury2014', 'ibims1', 'hammer', 'bokeh_failure', 'sintel', 'sintel_final'], default='vkitti2',
                        help='Dataset to process: vkitti2, kitti_test, kitti_train, nyuv2, eth3d, hypersim, make3d_kaggle, make3d_official, opensun3d, middlebury2014, ibims1, hammer, bokeh_failure, sintel or sintel_final')
    # Dataset paths (reusing the KITTI2 layout)
    parser.add_argument('--rgb_root', type=str, default='/mnt/slurm_home/hwzhang/Depth_Dataset/Virtual_KITTI_2/rgb')
    parser.add_argument('--depth_root', type=str, default='/mnt/slurm_home/hwzhang/Depth_Dataset/Virtual_KITTI_2/depth')
    parser.add_argument('--output_root', type=str, default='/mnt/slurm_home/hwzhang/Depth_Dataset/Virtual_KITTI_2/bokeh_diffusion_defocus_stack')
    parser.add_argument('--vkitti2_cameras', type=str, nargs='+', default=['Camera_0'],
                        help='Virtual KITTI 2 camera selection. Defaults to Camera_0 only; pass "all" to enable every camera.')

    parser.add_argument('--kitti_data_root', type=str,
                        default='/mnt/slurm_home/hwzhang/Depth_Dataset/kitti/monodepth2/kitti_data',
                        help='KITTI data root (used when --dataset is kitti_test or kitti_train)')
    parser.add_argument('--kitti_test_files', type=str,
                        default='/mnt/slurm_home/hwzhang/Depth_Dataset/kitti/monodepth2/splits/eigen_benchmark/test_files.txt',
                        help='KITTI sample list file (only used when --dataset kitti_test)')
    parser.add_argument('--kitti_train_files', type=str,
                        default='/mnt/slurm_home/hwzhang/Depth_Dataset/kitti/monodepth2/splits/eigen_zhou/train_files.txt',
                        help='KITTI training sample list file (only used when --dataset kitti_train)')
    parser.add_argument('--kitti_train_depths_root', type=str,
                        default='/mnt/slurm_home/hwzhang/Depth_Dataset/kitti/monodepth2/kitti_data/eigen_train_bokeh',
                        help='Existing KITTI training depth directory (e.g. depth.npy produced by bokehme; used when --dataset kitti_train)')
    parser.add_argument('--kitti_gt_depths', type=str,
                        default='/mnt/slurm_home/hwzhang/Depth_Dataset/kitti/monodepth2/splits/eigen_benchmark/gt_depths.npz',
                        help='KITTI depth ground truth (optional, used only when --dataset kitti_test)')
    parser.add_argument('--nyuv2_root', type=str,
                        default='/mnt/slurm_home/hwzhang/Depth_Dataset/nyuv2_official',
                        help='NYUv2 official-split data root (only used when --dataset nyuv2)')
    parser.add_argument('--eth3d_root', type=str,
                        default='/mnt/slurm_home/hwzhang/Depth_Dataset/eth3d_hr_mvs_raw',
                        help='ETH3D high-res MVS data root (only used when --dataset eth3d)')
    parser.add_argument('--make3d_kaggle_root', type=str,
                        default='/mnt/slurm_home/hwzhang/Depth_Dataset/make3d_kaggle/saxena_monocular_depth_2',
                        help='Make3D Kaggle data root (only used when --dataset make3d_kaggle)')
    parser.add_argument('--make3d_official_root', type=str,
                        default='/mnt/slurm_home/hwzhang/Depth_Dataset/make3d_offical',
                        help='Make3D Official (Test134) data root (only used when --dataset make3d_official)')
    # Sintel Depth dataset root (contains training/depth, training/camdata_left, training/depth_viz)
    parser.add_argument('--sintel_root', type=str,
                        default='/mnt/slurm_home/hwzhang/Depth_Dataset/SintelDepth',
                        help='Sintel Depth dataset root (contains the training/ subdirectory)')
    parser.add_argument('--opensun3d_root', type=str,
                        default='/mnt/slurm_home/hwzhang/Depth_Dataset/OpenSUN3D/dev',
                        help='OpenSUN3D data root (dev/test) (only used when --dataset opensun3d)')
    parser.add_argument('--hypersim_root', type=str,
                        default='/mnt/slurm_home/hwzhang/Depth_Dataset/ml-hypersim/hypersim_minimal',
                        help='Hypersim minimal root (only used when --dataset hypersim)')
    parser.add_argument('--hypersim_split_csv', type=str,
                        default='/mnt/slurm_home/hwzhang/Depth_Dataset/ml-hypersim/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv',
                        help='Hypersim split metadata CSV path')
    parser.add_argument('--hypersim_split', type=str, default='all',
                        help='Hypersim split name (train/val/test/all; "all" processes every split)')
    parser.add_argument('--middlebury_root', type=str,
                        default='/mnt/slurm_home/hwzhang/Depth_Dataset/Middlebury2014',
                        help='Middlebury Stereo 2014 dataset root (only used when --dataset middlebury2014)')
    parser.add_argument('--ibims1_root', type=str,
                        default='/mnt/slurm_home/hwzhang/Depth_Dataset/IBIMS1',
                        help='iBims-1 dataset root (only used when --dataset ibims1)')
    parser.add_argument('--hammer_root', type=str,
                        default='/mnt/slurm_home/hwzhang/Depth_Dataset/HAMMER/hammer_val',
                        help='HAMMER validation-set root (only used when --dataset hammer)')
    parser.add_argument('--hammer_depth_type', type=str,
                        default='depth_l515',
                        choices=['depth_l515', 'depth_d435', 'depth_tof'],
                        help='HAMMER depth modality (depth_l515/depth_d435/depth_tof, only used when --dataset hammer)')
    parser.add_argument('--bokeh_failure_manifest', type=str,
                        default='/mnt/slurm_home/hwzhang/BokehDepth/bokeh-generation/dataset/bokeh_failure_depthpro/bokeh_failure_top1200_depthpro.jsonl',
                        help='bokeh_failure dataset manifest (JSONL)')
    parser.add_argument('--bokeh_failure_root', type=str,
                        default='/mnt/slurm_home/hwzhang/BokehDepth/bokeh-generation/dataset/bokeh_failure_depthpro',
                        help='bokeh_failure dataset root')

    # Image size and defocus-stack configuration
    parser.add_argument('--assigned_size', type=int, default=None, help='Fixed evaluation size; None means use a VAE-aligned adaptive size')
    parser.add_argument('--k_min', type=float, default=1.0)
    parser.add_argument('--k_max', type=float, default=30.0)
    parser.add_argument('--k_step', type=float, default=5.0)
    parser.add_argument('--resize', action='store_true', help='Before generation, resize the short side to the specified size (keeping aspect ratio)')
    parser.add_argument('--size', type=int, default=518, help='Target short side when --resize is enabled')

    # I2I generation configuration
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_template", type=str, default=None, help='Prompt template; if None the default dof_cond template is used')

    # K normalization
    parser.add_argument("--K_max", type=float, default=30.0, help="Maximum value used to normalize K")

    # Misc
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process; None means all")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--inverse", action='store_true', help="Generate bokeh images from the end of the K sequence backwards")

    default_output_root = parser.get_default('output_root')
    args = parser.parse_args()
    args.blocks = [FLUX_TRANSFORMER_BLOCKS[int(i)] for i in args.block_ids]

    # Normalize the Virtual KITTI 2 camera selection.
    vkitti2_cameras: Optional[Set[str]]
    if args.vkitti2_cameras is None:
        vkitti2_cameras = None
    else:
        normalized: Set[str] = set()
        for cam in args.vkitti2_cameras:
            cam_norm = cam.strip()
            if not cam_norm:
                continue
            if cam_norm.lower() == 'all':
                normalized.clear()
                vkitti2_cameras = None
                break
            if not cam_norm.startswith('Camera_'):
                cam_norm = f'Camera_{cam_norm}'
            normalized.add(cam_norm)
        else:
            vkitti2_cameras = normalized
    args.vkitti2_cameras = vkitti2_cameras

    # Normalize the Hypersim split arg: "all" or empty string means every split.
    if args.dataset == 'hypersim':
        hyp_split = (args.hypersim_split or '').strip()
        if not hyp_split:
            hyp_split = 'all'
        args.hypersim_split = hyp_split

    if args.dataset == 'kitti_test' and args.output_root == default_output_root:
        args.output_root = os.path.join(args.kitti_data_root, 'bokeh_diffusion_defocus_stack')
    elif args.dataset == 'nyuv2' and args.output_root == default_output_root:
        args.output_root = os.path.join(args.nyuv2_root, 'bokeh_diffusion_defocus_stack')
    elif args.dataset == 'eth3d' and args.output_root == default_output_root:
        args.output_root = os.path.join(args.eth3d_root, 'bokeh_diffusion_defocus_stack')
    elif args.dataset == 'hypersim' and args.output_root == default_output_root:
        args.output_root = os.path.join(args.hypersim_root, 'bokeh_diffusion_defocus_stack')
    elif args.dataset == 'make3d_kaggle' and args.output_root == default_output_root:
        args.output_root = os.path.join(args.make3d_kaggle_root, 'bokeh_diffusion_defocus_stack')
    elif args.dataset == 'make3d_official' and args.output_root == default_output_root:
        args.output_root = os.path.join(args.make3d_official_root, 'bokeh_depth_defocus_stack')
    elif args.dataset == 'opensun3d' and args.output_root == default_output_root:
        args.output_root = os.path.join(args.opensun3d_root, 'bokeh_diffusion_defocus_stack')
    elif args.dataset == 'middlebury2014' and args.output_root == default_output_root:
        args.output_root = os.path.join(args.middlebury_root, 'bokeh_diffusion_defocus_stack')
    elif args.dataset == 'ibims1' and args.output_root == default_output_root:
        args.output_root = os.path.join(args.ibims1_root, 'bokeh_diffusion_defocus_stack')
    elif args.dataset == 'hammer' and args.output_root == default_output_root:
        args.output_root = os.path.join(args.hammer_root, 'bokeh_diffusion_defocus_stack')
    elif args.dataset == 'bokeh_failure' and args.output_root == default_output_root:
        args.output_root = args.bokeh_failure_root
    elif args.dataset in {'sintel', 'sintel_final'} and args.output_root == default_output_root:
        # Accept sintel_root pointing at either SintelDepth or SintelDepth/training.
        tr_root = args.sintel_root
        if os.path.isdir(os.path.join(tr_root, 'training')):
            tr_root = os.path.join(tr_root, 'training')
        variant_suffix = 'clean' if args.dataset == 'sintel' else 'final'
        args.output_root = os.path.join(tr_root, f'bokeh_depth_defocus_stack_{variant_suffix}')
    return args


def center_crop_to_size(arr: np.ndarray, target_h: Optional[int], target_w: Optional[int]) -> np.ndarray:
    """Center-crop to the specified size (reused from the KITTI2 code)."""
    if target_h is None or target_w is None:
        return arr
    if arr.ndim == 2:
        h, w = arr.shape
        c = arr
    else:
        h, w, _ = arr.shape
        c = arr
    if h < target_h or w < target_w:
        raise ValueError(f"Input smaller than target crop: input=({h},{w}) target=({target_h},{target_w})")
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    if arr.ndim == 2:
        return c[top: top + target_h, left: left + target_w]
    return c[top: top + target_h, left: left + target_w, :]


def load_depth_any(path: str) -> np.ndarray:
    """Load a depth map in an arbitrary format (reused from the KITTI2 code)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        depth = np.load(path)
    elif ext == '.npz':
        data = np.load(path)
        if 'depth' in data:
            depth = data['depth']
        else:
            # Take the first array.
            key0 = list(data.keys())[0]
            depth = data[key0]
    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Failed to read depth: {path}")
        depth = depth.astype(np.float32)
    return depth.astype(np.float32)


def load_sintel_depth(dpt_path: str) -> np.ndarray:
    """Read a Sintel Depth .dpt file into a float32 numpy array.

    File format: TAG_FLOAT(202021.25, float32) + width(int32) + height(int32) + depth(height*width float32)
    """
    with open(dpt_path, 'rb') as f:
        tag = np.fromfile(f, dtype=np.float32, count=1)
        if tag.size != 1 or abs(float(tag[0]) - 202021.25) > 1e-3:
            raise ValueError(f"Wrong tag in Sintel depth file: {dpt_path}")
        width = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        height = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        if width <= 0 or height <= 0 or width * height > 100000000:
            raise ValueError(f"Invalid Sintel depth size: {width}x{height} in {dpt_path}")
        depth = np.fromfile(f, dtype=np.float32, count=width * height)
        if depth.size != width * height:
            raise ValueError(f"Unexpected EOF in {dpt_path}")
    return depth.reshape((height, width)).astype(np.float32)



def find_matching_depth(depth_root: str, rel_scene: str, stem: str) -> Optional[str]:
    """Locate the matching depth file (reused from the KITTI2 code)."""
    # Split scene path and camera dir.
    parts = rel_scene.split(os.sep)
    if len(parts) == 0:
        return None
    camera_dir = parts[-1]
    scene_dir = os.path.join(*parts[:-1]) if len(parts) > 1 else ''

    # Expected depth directory.
    depth_dir = os.path.join(depth_root, scene_dir, 'frames', 'depth', camera_dir)
    if not os.path.isdir(depth_dir):
        return None

    # Prefer the Virtual KITTI 2 standard name.
    preferred = os.path.join(depth_dir, f'depth_{stem}.png')
    if os.path.exists(preferred):
        return preferred

    # Tolerate a few alternative extensions.
    for ext in ['.png', '.npy', '.npz']:
        cand = os.path.join(depth_dir, f'depth_{stem}{ext}')
        if os.path.exists(cand):
            return cand

    # Loose match as a last resort.
    try:
        for fname in os.listdir(depth_dir):
            name_wo_ext, _ = os.path.splitext(fname)
            if name_wo_ext.lower() == f'depth_{stem}'.lower():
                return os.path.join(depth_dir, fname)
    except Exception:
        pass
    return None


def list_vkitti2_images(root: str, allowed_cameras: Optional[Set[str]] = None) -> List[Tuple[str, str, str]]:
    """Recursively collect Virtual KITTI 2 RGB images (reused from the KITTI2 code).

    Args:
        root: Virtual KITTI 2 RGB root directory.
        allowed_cameras: Set of camera directory names to keep (e.g. {"Camera_0"});
            None means no filtering.
    """
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    out: List[Tuple[str, str, str]] = []

    if allowed_cameras is not None and len(allowed_cameras) == 0:
        return out

    for dirpath, _, filenames in os.walk(root):
        # Keep only directories that contain .../frames/rgb/Camera_*
        pattern = f"{os.sep}frames{os.sep}rgb{os.sep}Camera_"
        if pattern not in dirpath:
            continue

        # Parse the relative path; extract scene and camera.
        rel_dir = os.path.relpath(dirpath, root)
        parts = rel_dir.split(os.sep)
        try:
            frames_idx = parts.index('frames')
        except ValueError:
            continue
        # Expected layout: [SceneX, Y, 'frames', 'rgb', 'Camera_Z']
        if len(parts) < frames_idx + 3:
            continue
        camera_dir = parts[frames_idx + 2]
        if not camera_dir.startswith('Camera_'):
            continue
        if allowed_cameras is not None and camera_dir not in allowed_cameras:
            continue
        scene_parts = parts[:frames_idx]  # [SceneX, Y]
        scene_rel = os.path.join(*scene_parts, camera_dir) if scene_parts else camera_dir

        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in exts:
                continue
            if not fname.startswith('rgb_'):
                continue
            name_wo_ext = os.path.splitext(fname)[0]  # rgb_00000
            frame_id = name_wo_ext.split('rgb_')[-1]
            abs_path = os.path.join(dirpath, fname)
            out.append((scene_rel, frame_id, abs_path))

    # Stable ordering: sort by scene_rel first, then by frame id.
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def list_nyuv2_images(nyuv2_root: str) -> List[Tuple[str, str, str, Optional[str]]]:
    """Collect the RGB/depth pairs from the NYUv2 official splits (train and test)."""
    items: List[Tuple[str, str, str, Optional[str]]] = []

    if not os.path.isdir(nyuv2_root):
        return items

    # Iterate splits to keep the output order stable.
    splits = []
    for split_name in ['train', 'test']:
        split_path = os.path.join(nyuv2_root, split_name)
        if os.path.isdir(split_path):
            splits.append((split_name, split_path))

    for _, split_path in splits:
        for dirpath, _, filenames in os.walk(split_path):
            if not filenames:
                continue

            # Build a one-shot lookup map.
            depth_candidates = {}
            for fname in filenames:
                if not fname.startswith('depth_'):
                    continue
                name_wo_ext, ext = os.path.splitext(fname)
                depth_candidates[name_wo_ext] = os.path.join(dirpath, fname)

            rgb_names = [fname for fname in filenames if fname.startswith('rgb_') and fname.lower().endswith('.png')]
            if not rgb_names:
                continue

            for rgb_name in sorted(rgb_names):
                stem = rgb_name.split('rgb_')[-1].split('.')[0]
                rgb_path = os.path.join(dirpath, rgb_name)

                depth_key = f'depth_{stem}'
                depth_path: Optional[str] = depth_candidates.get(depth_key)

                if depth_path is None:
                    # Fall back to checking common extensions.
                    for ext in ('.png', '.npy', '.npz'):
                        cand = os.path.join(dirpath, f'{depth_key}{ext}')
                        if os.path.exists(cand):
                            depth_path = cand
                            break

                rel_dir = os.path.relpath(dirpath, nyuv2_root)
                items.append((rel_dir, stem, rgb_path, depth_path))

    items.sort(key=lambda x: (x[0], x[1]))
    return items


def list_sintel_images(sintel_root: str) -> List[Tuple[str, str, str, str, Optional[str], Optional[str]]]:
    """Collect clean/final RGB, depth and camera paths from the MPI-Sintel Depth training set.

    Returned tuples: (scene, frame_stem, clean_rgb_path, depth_dpt_path, cam_path, final_rgb_path)

    Accepts two root layouts:
    - Pointing at SintelDepth/ (contains a training/ subdirectory)
    - Pointing directly at SintelDepth/training/
    """
    items: List[Tuple[str, str, str, str, Optional[str], Optional[str]]] = []
    tr_root = sintel_root
    if os.path.isdir(os.path.join(tr_root, 'training')):
        tr_root = os.path.join(tr_root, 'training')

    depth_root = os.path.join(tr_root, 'depth')
    clean_root = os.path.join(tr_root, 'clean')
    final_root = os.path.join(tr_root, 'final')
    cam_root = os.path.join(tr_root, 'camdata_left')

    if not os.path.isdir(depth_root):
        raise FileNotFoundError(f'Sintel depth directory not found: {depth_root}')
    if not os.path.isdir(clean_root):
        raise FileNotFoundError(f'Sintel clean RGB directory not found: {clean_root}')
    if not os.path.isdir(final_root):
        raise FileNotFoundError(f'Sintel final RGB directory not found: {final_root}')

    for scene in sorted(os.listdir(clean_root)):
        clean_scene_dir = os.path.join(clean_root, scene)
        depth_scene_dir = os.path.join(depth_root, scene)
        if not os.path.isdir(clean_scene_dir) or not os.path.isdir(depth_scene_dir):
            continue

        final_scene_dir = os.path.join(final_root, scene)
        cam_scene_dir = os.path.join(cam_root, scene)

        for fname in sorted(os.listdir(clean_scene_dir)):
            if not fname.startswith('frame_') or not fname.lower().endswith('.png'):
                continue

            stem = os.path.splitext(fname)[0]
            clean_path = os.path.join(clean_scene_dir, fname)
            depth_path = os.path.join(depth_scene_dir, f'{stem}.dpt')
            if not os.path.isfile(depth_path):
                continue

            final_path: Optional[str] = None
            if os.path.isdir(final_scene_dir):
                candidate_final = os.path.join(final_scene_dir, fname)
                if os.path.isfile(candidate_final):
                    final_path = candidate_final
            if final_path is None:
                continue

            cam_path: Optional[str] = None
            if os.path.isdir(cam_scene_dir):
                candidate_cam = os.path.join(cam_scene_dir, f'{stem}.cam')
                if os.path.isfile(candidate_cam):
                    cam_path = candidate_cam

            items.append((scene, stem, clean_path, depth_path, cam_path, final_path))

    items.sort(key=lambda x: (x[0], x[1]))
    return items


def list_eth3d_images(eth3d_root: str) -> List[Tuple[str, str, str, Optional[str], Optional[str]]]:
    """Collect RGB / depth / mask paths from the ETH3D high-res MVS dataset."""
    items: List[Tuple[str, str, str, Optional[str], Optional[str]]] = []

    img_root = os.path.join(eth3d_root, 'img')
    if not os.path.isdir(img_root):
        return items

    scene_names = [d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))]
    for scene in sorted(scene_names):
        rgb_dir = os.path.join(img_root, scene, 'images', 'dslr_images')
        if not os.path.isdir(rgb_dir):
            continue

        depth_dir = os.path.join(eth3d_root, 'depth', scene, 'ground_truth_depth', 'dslr_images')
        mask_dir = os.path.join(eth3d_root, scene, 'masks_for_images', 'dslr_images')

        rgb_filenames = sorted(os.listdir(rgb_dir))
        for fname in rgb_filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in {'.jpg', '.jpeg', '.png'}:
                continue

            rgb_path = os.path.join(rgb_dir, fname)
            stem = os.path.splitext(fname)[0]

            depth_path = os.path.join(depth_dir, fname)
            if not os.path.exists(depth_path):
                depth_path = None
                for alt_ext in ('.pfm', '.bin', '.png', '.npy', '.npz'):
                    candidate = os.path.join(depth_dir, f'{stem}{alt_ext}')
                    if os.path.exists(candidate):
                        depth_path = candidate
                        break
            if depth_path is None or not os.path.exists(depth_path):
                continue

            mask_path: Optional[str] = None
            if os.path.isdir(mask_dir):
                for mask_ext in ('.png', '.PNG', '.jpg', '.JPG'):
                    candidate = os.path.join(mask_dir, f'{stem}{mask_ext}')
                    if os.path.exists(candidate):
                        mask_path = candidate
                        break

            items.append((scene, stem, rgb_path, depth_path, mask_path))

    items.sort(key=lambda x: (x[0], x[1]))
    return items


def list_make3d_images(make3d_root: str) -> List[Tuple[str, str, str, str]]:
    """Collect RGB / depth paths from the Make3D (Saxena et al.) dataset."""
    items: List[Tuple[str, str, str, str]] = []

    if not os.path.isdir(make3d_root):
        return items

    rgb_exts = {'.jpg', '.jpeg', '.png'}
    filenames = sorted(os.listdir(make3d_root))
    for fname in filenames:
        ext = os.path.splitext(fname)[1].lower()
        if ext not in rgb_exts:
            continue

        stem = os.path.splitext(fname)[0]
        depth_path = os.path.join(make3d_root, f'{stem}.dat')
        if not os.path.exists(depth_path):
            continue

        category, sep, remainder = stem.partition('-')
        if sep == '':
            category = 'default'
        rgb_path = os.path.join(make3d_root, fname)
        items.append((category, stem, rgb_path, depth_path))

    items.sort(key=lambda x: (x[0], x[1]))
    return items


def list_make3d_official_images(make3d_official_root: str) -> List[Tuple[str, str, str, str]]:
    """Collect RGB / depth paths from the Make3D Official (Test134) dataset.

    Args:
        make3d_official_root: dataset root, should contain Test134/ and Gridlaserdata/ subdirectories.

    Returns:
        List of (category, stem, rgb_path, depth_path)
        - category: category derived from scene_id (e.g. '01' for outdoor, '02' for buildings, ...)
        - stem: full identifier such as '01-p-bfhgh'
        - rgb_path: Test134/img-01-p-bfhgh.jpg
        - depth_path: Gridlaserdata/depth_sph_corr-01-p-bfhgh.mat
    """
    items: List[Tuple[str, str, str, str]] = []

    if not os.path.isdir(make3d_official_root):
        return items

    # Verify required subdirectories.
    rgb_dir = os.path.join(make3d_official_root, 'Test134')
    depth_dir = os.path.join(make3d_official_root, 'Gridlaserdata')

    if not os.path.isdir(rgb_dir):
        print(f'[WARN] Make3D Official RGB directory not found: {rgb_dir}')
        return items

    if not os.path.isdir(depth_dir):
        print(f'[WARN] Make3D Official depth directory not found: {depth_dir}')
        return items

    # Iterate RGB images under Test134.
    rgb_exts = {'.jpg', '.jpeg', '.JPG', '.JPEG'}
    filenames = sorted(os.listdir(rgb_dir))

    for fname in filenames:
        ext = os.path.splitext(fname)[1]
        if ext not in rgb_exts:
            continue

        # Filename format: img-{scene_id}-p-{view_id}.jpg
        # e.g. img-01-p-bfhgh.jpg
        if not fname.startswith('img-'):
            continue

        # Extract the stem: drop the 'img-' prefix and the extension.
        stem_with_prefix = os.path.splitext(fname)[0]  # img-01-p-bfhgh
        stem = stem_with_prefix[4:]  # 01-p-bfhgh (drop the 'img-' prefix)

        # Build the matching depth-file path.
        depth_fname = f'depth_sph_corr-{stem}.mat'
        depth_path = os.path.join(depth_dir, depth_fname)

        if not os.path.exists(depth_path):
            print(f'[WARN] Make3D Official depth file not found: {depth_path}')
            continue

        # Derive the category (first two digits of scene_id, e.g. '01').
        category_parts = stem.split('-')
        category = category_parts[0] if category_parts else 'unknown'

        rgb_path = os.path.join(rgb_dir, fname)
        items.append((category, stem, rgb_path, depth_path))

    items.sort(key=lambda x: (x[0], x[1]))
    return items


def list_opensun3d_frames(opensun3d_root: str) -> List[Tuple[str, str, str, str, str, Optional[str]]]:
    """Collect lowres-wide RGB / depth paths and intrinsics from OpenSUN3D."""
    items: List[Tuple[str, str, str, str, str, Optional[str]]] = []

    if not os.path.isdir(opensun3d_root):
        return items

    for dirpath, dirnames, _ in os.walk(opensun3d_root):
        required = {'lowres_wide', 'lowres_depth'}
        if not required.issubset(set(dirnames)):
            continue

        rel_path = os.path.relpath(dirpath, opensun3d_root)
        parts = rel_path.split(os.sep)
        if len(parts) == 1:
            scene_rel = ''
            sequence_id = parts[0]
        else:
            scene_rel = os.path.join(*parts[:-1])
            sequence_id = parts[-1]

        rgb_dir = os.path.join(dirpath, 'lowres_wide')
        depth_dir = os.path.join(dirpath, 'lowres_depth')
        intr_dir = os.path.join(dirpath, 'lowres_wide_intrinsics')

        rgb_files = {f for f in os.listdir(rgb_dir) if f.lower().endswith('.png')}
        depth_files = {f for f in os.listdir(depth_dir) if f.lower().endswith('.png')}
        common_files = sorted(rgb_files & depth_files)
        if not common_files:
            continue

        intrinsics_available = set()
        if os.path.isdir(intr_dir):
            intrinsics_available = {f for f in os.listdir(intr_dir) if f.lower().endswith('.pincam')}

        for filename in common_files:
            stem, _ = os.path.splitext(filename)
            rgb_path = os.path.join(rgb_dir, filename)
            depth_path = os.path.join(depth_dir, filename)
            intrinsics_path: Optional[str] = None
            intr_name = f'{stem}.pincam'
            if intr_name in intrinsics_available:
                intrinsics_path = os.path.join(intr_dir, intr_name)

            items.append((scene_rel, sequence_id, stem, rgb_path, depth_path, intrinsics_path))

    items.sort(key=lambda x: (x[0], x[1], x[2]))
    return items


def list_ibims1_images(
    ibims1_root: str,
) -> List[Tuple[str, str, str, Optional[str], Optional[str], Optional[str]]]:
    """Collect RGB / depth / mask / camera-intrinsics paths from iBims-1."""
    items: List[Tuple[str, str, str, Optional[str], Optional[str], Optional[str]]] = []

    core_root = os.path.join(ibims1_root, 'core_raw', 'ibims1_core_raw')
    rgb_dir = os.path.join(core_root, 'rgb')
    depth_dir = os.path.join(core_root, 'depth')
    mask_invalid_dir = os.path.join(core_root, 'mask_invalid')
    mask_transp_dir = os.path.join(core_root, 'mask_transp')
    calib_dir = os.path.join(core_root, 'calib')

    if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir):
        print(f'[WARN] iBims-1 directory missing: rgb_dir={rgb_dir}, depth_dir={depth_dir}')
        return items

    imagelist: List[str] = []
    for candidate in [
        os.path.join(ibims1_root, 'imagelist.txt'),
        os.path.join(core_root, 'imagelist.txt'),
        os.path.join(ibims1_root, 'eval_list.txt'),
    ]:
        if os.path.isfile(candidate):
            with open(candidate, 'r', encoding='utf-8') as f:
                imagelist = [line.strip() for line in f if line.strip()]
            if imagelist:
                break

    if not imagelist:
        imagelist = [
            os.path.splitext(fname)[0]
            for fname in os.listdir(rgb_dir)
            if fname.lower().endswith('.png')
        ]

    for scene_name in imagelist:
        rgb_path = os.path.join(rgb_dir, f'{scene_name}.png')
        depth_path = os.path.join(depth_dir, f'{scene_name}.png')
        if not os.path.exists(rgb_path):
            print(f'[WARN] iBims-1 missing RGB: {scene_name}')
            continue
        if not os.path.exists(depth_path):
            print(f'[WARN] iBims-1 missing depth: {scene_name}')
            continue

        mask_invalid_path: Optional[str] = None
        cand_invalid = os.path.join(mask_invalid_dir, f'{scene_name}.png')
        if os.path.exists(cand_invalid):
            mask_invalid_path = cand_invalid

        mask_transp_path: Optional[str] = None
        cand_transp = os.path.join(mask_transp_dir, f'{scene_name}.png')
        if os.path.exists(cand_transp):
            mask_transp_path = cand_transp

        calib_path: Optional[str] = None
        cand_calib = os.path.join(calib_dir, f'{scene_name}.txt')
        if os.path.exists(cand_calib):
            calib_path = cand_calib

        items.append((scene_name, rgb_path, depth_path, mask_invalid_path, mask_transp_path, calib_path))

    return items


def list_middlebury2014_images(
    middlebury_root: str,
) -> List[Tuple[str, str, str, str, float, float, Optional[str], Optional[str]]]:
    """Collect im0 / disp0 / calib paths and calibration parameters from Middlebury Stereo 2014 (Perfect)."""
    items: List[Tuple[str, str, str, str, float, float, Optional[str], Optional[str]]] = []

    dataset_root = os.path.join(middlebury_root, 'datasets')
    if not os.path.isdir(dataset_root):
        print(f'[WARN] Middlebury2014 root missing or has no datasets/: {dataset_root}')
        return items

    available_dirs = {
        name: os.path.join(dataset_root, name)
        for name in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, name))
    }

    ordered_dirs: List[str] = []
    imagelist_path = os.path.join(middlebury_root, 'imagelist.txt')
    if os.path.isfile(imagelist_path):
        with open(imagelist_path, 'r', encoding='utf-8') as f:
            for raw in f:
                entry = raw.strip()
                if not entry:
                    continue
                candidates = [entry, f'{entry}-perfect']
                for cand in candidates:
                    if cand in available_dirs and cand not in ordered_dirs:
                        ordered_dirs.append(cand)
                        break
    for dir_name in sorted(available_dirs.keys()):
        if dir_name not in ordered_dirs:
            ordered_dirs.append(dir_name)

    for dir_name in ordered_dirs:
        scene_dir = available_dirs.get(dir_name)
        if scene_dir is None:
            continue
        scene_name = dir_name.replace('-perfect', '')
        rgb_path = os.path.join(scene_dir, 'im0.png')
        disp_path = os.path.join(scene_dir, 'disp0.pfm')
        calib_path = os.path.join(scene_dir, 'calib.txt')
        if not (os.path.exists(rgb_path) and os.path.exists(disp_path) and os.path.exists(calib_path)):
            print(f'[WARN] Middlebury2014 missing required files: {dir_name}')
            continue

        baseline_m, focal_px = parse_middlebury_calib(calib_path)
        if baseline_m is None or focal_px is None:
            print(f'[WARN] Middlebury2014 calibration info missing: {calib_path}')
            continue

        samples_path = os.path.join(scene_dir, 'disp0-n.pgm')
        sd_path = os.path.join(scene_dir, 'disp0-sd.pfm')
        items.append(
            (
                scene_name,
                rgb_path,
                disp_path,
                calib_path,
                baseline_m,
                float(focal_px),
                samples_path if os.path.exists(samples_path) else None,
                sd_path if os.path.exists(sd_path) else None,
            )
        )

    return items


def list_hammer_images(hammer_root: str, depth_type: str) -> List[Tuple[str, str, str, str, str, Optional[str]]]:
    """Collect RGB / depth paths from the HAMMER validation set."""
    items: List[Tuple[str, str, str, str, str, Optional[str]]] = []

    if not os.path.isdir(hammer_root):
        print(f'[WARN] HAMMER root not found: {hammer_root}')
        return items

    sensor_dir = 'polarization'
    depth_dir_name = depth_type

    for scene_name in sorted(os.listdir(hammer_root)):
        scene_path = os.path.join(hammer_root, scene_name)
        if not os.path.isdir(scene_path):
            continue

        polar_dir = os.path.join(scene_path, sensor_dir)
        rgb_dir = os.path.join(polar_dir, 'rgb')
        depth_dir = os.path.join(polar_dir, depth_dir_name)

        if not os.path.isdir(rgb_dir):
            continue
        if not os.path.isdir(depth_dir):
            print(f'[WARN] HAMMER missing depth directory: {scene_name}/{sensor_dir}/{depth_dir_name}')
            continue

        intrinsics_path = os.path.join(polar_dir, 'intrinsics.txt')
        if not os.path.isfile(intrinsics_path):
            intrinsics_path = None

        for fname in sorted(os.listdir(rgb_dir)):
            if not fname.lower().endswith('.png'):
                continue
            rgb_path = os.path.join(rgb_dir, fname)
            depth_path = os.path.join(depth_dir, fname)
            if not os.path.exists(depth_path):
                print(f'[WARN] HAMMER missing depth file: {scene_name}/{sensor_dir}/{depth_dir_name}/{fname}')
                continue
            stem = os.path.splitext(fname)[0]
            items.append((scene_name, sensor_dir, depth_dir_name, stem, rgb_path, depth_path, intrinsics_path))

    items.sort(key=lambda x: (x[0], x[3]))
    return items


def load_bokeh_failure_manifest(manifest_path: str, dataset_root: str) -> List[Dict[str, Any]]:
    """Read the JSONL manifest of the bokeh_failure dataset."""
    items: List[Dict[str, Any]] = []

    if not os.path.isfile(manifest_path):
        print(f'[WARN] bokeh_failure manifest not found: {manifest_path}')
        return items

    dataset_root_abs = os.path.abspath(dataset_root)

    with open(manifest_path, 'r') as f:
        for line_idx, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f'[WARN] failed to parse bokeh_failure manifest (line {line_idx}): {exc}')
                continue

            sample_dir_raw = record.get('sample_dir') or record.get('source_sample_dir')
            if not sample_dir_raw:
                print(f'[WARN] bokeh_failure manifest missing sample_dir (line {line_idx}), skipping')
                continue

            sample_dir_abs = os.path.abspath(sample_dir_raw)
            all_in_focus_path = record.get('all_in_focus_path')
            if not all_in_focus_path:
                candidate = os.path.join(sample_dir_abs, 'all_in_focus.png')
                if os.path.exists(candidate):
                    all_in_focus_path = candidate
            if not all_in_focus_path or not os.path.exists(all_in_focus_path):
                print(f'[WARN] all_in_focus image not found (line {line_idx}): {all_in_focus_path}')
                continue

            depth_path = record.get('depth_path') or record.get('original_depth_path')
            if not depth_path:
                candidate_depth = os.path.join(sample_dir_abs, 'depth.npy')
                if os.path.exists(candidate_depth):
                    depth_path = candidate_depth
            if not depth_path or not os.path.exists(depth_path):
                print(f'[WARN] depth file not found (line {line_idx}): {depth_path}')
                continue

            original_depth_path = record.get('original_depth_path')
            similarity_score = record.get('similarity_score')

            rel_path: str
            sample_dir_rel: Optional[str] = None
            try:
                if os.path.commonpath([dataset_root_abs, sample_dir_abs]) == dataset_root_abs:
                    sample_dir_rel = os.path.relpath(sample_dir_abs, dataset_root_abs)
            except ValueError:
                sample_dir_rel = None
            rel_path = sample_dir_rel if sample_dir_rel is not None else os.path.basename(sample_dir_abs)

            items.append({
                'rel_path': rel_path,
                'sample_dir_abs': sample_dir_abs,
                'all_in_focus_path': all_in_focus_path,
                'depth_path': depth_path,
                'original_depth_path': original_depth_path,
                'similarity_score': similarity_score,
            })

    items.sort(key=lambda x: x['rel_path'])
    return items


def load_eth3d_depth(depth_path: str, image_size: Tuple[int, int]) -> np.ndarray:
    """Load an ETH3D depth map (float32 binary) and reshape it to H x W."""
    width, height = image_size
    expected = width * height
    depth_flat = np.fromfile(depth_path, dtype=np.float32)
    if depth_flat.size != expected:
        raise ValueError(
            f'ETH3D depth size mismatch: expected {expected} elements, got {depth_flat.size} ({depth_path})'
        )
    depth = depth_flat.reshape((height, width))
    return depth.astype(np.float32)


def load_make3d_depth(depth_path: str) -> np.ndarray:
    """Load a Make3D depth map as float32 (ASCII .dat)."""
    try:
        depth = np.loadtxt(depth_path, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f'Failed to read Make3D depth file: {depth_path}') from exc

    if depth.ndim == 0:
        depth = np.array([[float(depth)]], dtype=np.float32)

    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return depth.astype(np.float32)


def load_make3d_official_depth(depth_path: str) -> np.ndarray:
    """Load the depth map of a Make3D Official sample (Position3DGrid inside the .mat file).

    Args:
        depth_path: .mat file path, e.g. depth_sph_corr-01-p-bfhgh.mat

    Returns:
        depth_map: (H, W) float32 array using the Euclidean distance d (55 x 305).

    Position3DGrid layout:
        - [:,:,0]: Y (vertical coordinate, meters)
        - [:,:,1]: X (horizontal coordinate, meters)
        - [:,:,2]: Z (projection depth, meters)
        - [:,:,3]: d (Euclidean distance, meters) <- we use this channel
    """
    try:
        import scipy.io
        mat_data = scipy.io.loadmat(depth_path)
    except Exception as exc:
        raise RuntimeError(f'Failed to read Make3D Official depth file: {depth_path}') from exc

    # Load Position3DGrid (55 x 305 x 4).
    if 'Position3DGrid' not in mat_data:
        raise ValueError(f'Make3D Official .mat file missing Position3DGrid field: {depth_path}')

    pos_grid = mat_data['Position3DGrid']  # (55, 305, 4)

    if pos_grid.ndim != 3 or pos_grid.shape[2] != 4:
        raise ValueError(
            f'Make3D Official Position3DGrid has wrong layout: expected (H, W, 4), got {pos_grid.shape}'
        )

    # Take the 4th channel (index=3): Euclidean distance.
    depth = pos_grid[:, :, 3].astype(np.float32)  # (55, 305)

    # Sanitize.
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    # Make3D Official depth is already in meters; no extra scaling needed.
    return depth


HYPERSIM_FOCAL_LENGTH = 886.81


def load_pfm(path: str) -> np.ndarray:
    """Load a PFM file (float32) and return an H x W or H x W x 3 array."""
    with open(path, 'rb') as f:
        header = f.readline().decode('ascii').rstrip()
        if header not in {'PF', 'Pf'}:
            raise ValueError(f'Invalid PFM header: {header}')

        dimensions_line = f.readline().decode('ascii').strip()
        while dimensions_line.startswith('#'):
            dimensions_line = f.readline().decode('ascii').strip()
        try:
            width_str, height_str = dimensions_line.split()
            width, height = int(width_str), int(height_str)
        except ValueError as exc:
            raise ValueError(f'Invalid PFM dimensions: {dimensions_line}') from exc

        scale_line = f.readline().decode('ascii').strip()
        try:
            scale = float(scale_line)
        except ValueError as exc:
            raise ValueError(f'Invalid PFM scale: {scale_line}') from exc

        endian = '<' if scale < 0 else '>'
        data = np.fromfile(f, f'{endian}f')
        channel_count = 3 if header == 'PF' else 1
        expected = width * height * channel_count
        if data.size != expected:
            raise ValueError(f'PFM size mismatch: expected {expected}, got {data.size} ({path})')
        data = data.reshape((height, width, channel_count) if channel_count > 1 else (height, width))
        if scale > 0:
            data = np.flipud(data)
        return data.astype(np.float32)


def parse_middlebury_calib(calib_path: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse a Middlebury calibration file; returns (baseline, focal_length): baseline in meters, focal length in pixels."""
    baseline_m: Optional[float] = None
    focal_px: Optional[float] = None

    try:
        with open(calib_path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or '=' not in line:
                    continue
                if line.startswith('cam0='):
                    matrix_str = line.split('=', 1)[1].strip()
                    matrix_str = matrix_str.strip('[]')
                    first_row = matrix_str.split(';')[0]
                    parts = first_row.strip().split()
                    if parts:
                        focal_px = float(parts[0])
                elif line.startswith('baseline='):
                    baseline_mm = float(line.split('=', 1)[1].strip())
                    baseline_m = baseline_mm / 1000.0
                if baseline_m is not None and focal_px is not None:
                    break
    except Exception as exc:
        print(f'[WARN] failed to parse Middlebury calibration {calib_path}: {exc}')

    return baseline_m, focal_px


def load_middlebury_depth(disparity_path: str, baseline_m: float, focal_length_px: float) -> np.ndarray:
    """Convert disparity + calibration into depth in meters."""
    disparity = load_pfm(disparity_path).astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = (baseline_m * focal_length_px) / disparity
    invalid_mask = ~np.isfinite(depth) | (disparity <= 0)
    depth[invalid_mask] = 0.0
    return depth.astype(np.float32)


def load_ibims1_depth(depth_path: str) -> np.ndarray:
    """Load an iBims-1 depth map and convert it to meters."""
    depth_uint16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_uint16 is None:
        raise FileNotFoundError(f'iBims-1 depth file failed to read: {depth_path}')
    depth = depth_uint16.astype(np.float32)
    # Official scaling: 0-50m range mapped to uint16.
    depth *= (50.0 / 65535.0)
    return depth


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {'true', '1', 'yes'}


def _format_hypersim_frame_token(frame_id: int) -> str:
    width = max(4, len(str(frame_id)))
    return f'frame.{frame_id:0{width}d}'


def _format_hypersim_frame_dir(frame_id: int) -> str:
    width = max(4, len(str(frame_id)))
    return f'frame_{frame_id:0{width}d}'


@lru_cache(maxsize=None)
def _hypersim_distance_scale(width: int, height: int) -> np.ndarray:
    xs = np.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width, dtype=np.float32)
    ys = np.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height, dtype=np.float32)
    grid_x = np.broadcast_to(xs.reshape(1, width), (height, width))
    grid_y = np.broadcast_to(ys.reshape(height, 1), (height, width))
    grid_z = np.full((height, width), HYPERSIM_FOCAL_LENGTH, dtype=np.float32)
    norms = np.sqrt(grid_x ** 2 + grid_y ** 2 + grid_z ** 2)
    scale = HYPERSIM_FOCAL_LENGTH / np.maximum(norms, 1e-6)
    return scale.astype(np.float32)


def load_hypersim_color(color_path: str) -> np.ndarray:
    with h5py.File(color_path, 'r') as f:
        data = f['dataset'][()].astype(np.float32)
    data = np.clip(data, 0.0, 1.0)
    return data


def load_hypersim_depth(depth_path: str) -> np.ndarray:
    with h5py.File(depth_path, 'r') as f:
        distance = f['dataset'][()].astype(np.float32)
    height, width = distance.shape
    scale = _hypersim_distance_scale(width, height)
    depth = distance * scale
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return depth.astype(np.float32)


def list_hypersim_images(
    hypersim_root: str,
    split_csv: str,
    split_name: str,
) -> List[Tuple[str, str, str, int, str, str]]:
    """List Hypersim samples.

    Returns:
        List elements are (split, scene, camera, frame_id, color_path, depth_path).
    """
    items: List[Tuple[str, str, str, int, str, str]] = []

    if not os.path.isdir(hypersim_root):
        return items
    if not os.path.isfile(split_csv):
        return items

    normalized_split = (split_name or '').strip()
    target_split: Optional[str]
    if not normalized_split or normalized_split.lower() in {'all', '*'}:
        target_split = None
    else:
        target_split = normalized_split

    try:
        with open(split_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                split = row.get('split_partition_name', '').strip()
                if target_split and split != target_split:
                    continue
                if row.get('included_in_public_release') and not _is_truthy(row['included_in_public_release']):
                    continue

                scene = row.get('scene_name', '').strip()
                camera = row.get('camera_name', '').strip()
                frame_id_str = row.get('frame_id', '').strip()

                if not scene or not camera or not frame_id_str:
                    continue
                try:
                    frame_id = int(frame_id_str)
                except ValueError:
                    continue

                camera_dir = f'scene_{camera}'
                frame_token = _format_hypersim_frame_token(frame_id)

                color_dir = os.path.join(hypersim_root, scene, 'images', f'{camera_dir}_final_hdf5')
                depth_dir = os.path.join(hypersim_root, scene, 'images', f'{camera_dir}_geometry_hdf5')
                color_path = os.path.join(color_dir, f'{frame_token}.color.hdf5')
                depth_path = os.path.join(depth_dir, f'{frame_token}.depth_meters.hdf5')

                if not os.path.exists(color_path) or not os.path.exists(depth_path):
                    continue
                split_normalized = split if split else 'unspecified'

                # Use absolute paths so the downstream manifest can reference the original HDF5 directly without copying.
                items.append(
                    (
                        split_normalized,
                        scene,
                        camera,
                        frame_id,
                        os.path.abspath(color_path),
                        os.path.abspath(depth_path),
                    )
                )
    except Exception as exc:
        print(f'[WARN] failed to read Hypersim split CSV: {split_csv}: {exc}')

    items.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return items


def ensure_dir(path: str) -> None:
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)


def check_sample_complete(sample_dir: str, expected_k_count: int) -> bool:
    """Check whether a sample has been fully generated.

    Args:
        sample_dir: sample output directory.
        expected_k_count: expected number of defocus-stack images.

    Returns:
        True if the sample is fully generated, False otherwise.
    """
    if not os.path.isdir(sample_dir):
        return False

    stack_dir = os.path.join(sample_dir, 'defocus_stack')
    if not os.path.isdir(stack_dir):
        return False

    # Ensure stack_index.json is present.
    meta_path = os.path.join(stack_dir, 'stack_index.json')
    if not os.path.isfile(meta_path):
        return False

    # Read the meta to verify the defocus-stack count.
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False

    ids = meta.get('ids') or []
    k_values = meta.get('k_values', {})

    if not ids and k_values:
        def _sort_key(token: str) -> Tuple[int, str]:
            try:
                return (0, int(token))
            except ValueError:
                return (1, token)

        ids = sorted([str(key) for key in k_values.keys()], key=_sort_key)

    if not ids:
        return False

    if len(ids) != expected_k_count:
        return False

    # Verify each defocus-stack image exists.
    for idx in ids:
        img_path = os.path.join(stack_dir, f'{idx}.png')
        if not os.path.isfile(img_path):
            return False

    return True


def generate_nyuv2_manifests(
    output_root: str,
    manifest_filename: str = 'manifest_bokeh_diffusion_nyuv2.jsonl',
) -> Dict[str, Tuple[str, int]]:
    """Scan NYUv2 outputs and write a JSONL manifest per split.

    Args:
        output_root: output root (contains train/test subdirectories).
        manifest_filename: filename of the JSONL to write.

    Returns:
        Mapping split -> (manifest_path, sample_count).
    """

    results: Dict[str, Tuple[str, int]] = {}
    if not os.path.isdir(output_root):
        print(f'[WARN] cannot generate NYUv2 manifest, output directory not found: {output_root}')
        return results

    for split_name in sorted(os.listdir(output_root)):
        split_dir = os.path.join(output_root, split_name)
        if not os.path.isdir(split_dir):
            continue

        manifest_entries: List[Dict[str, Any]] = []
        skipped_incomplete = 0

        for scene_name in sorted(os.listdir(split_dir)):
            scene_dir = os.path.join(split_dir, scene_name)
            if not os.path.isdir(scene_dir):
                continue

            for sample_name in sorted(os.listdir(scene_dir)):
                sample_dir = os.path.join(scene_dir, sample_name)
                if not os.path.isdir(sample_dir):
                    continue

                stack_dir = os.path.join(sample_dir, 'defocus_stack')
                stack_index_path = os.path.join(stack_dir, 'stack_index.json')

                if not (os.path.isdir(stack_dir) and os.path.isfile(stack_index_path)):
                    skipped_incomplete += 1
                    continue

                try:
                    with open(stack_index_path, 'r', encoding='utf-8') as f:
                        stack_info = json.load(f)
                except json.JSONDecodeError as exc:
                    print(f'[WARN] failed to parse stack_index ({stack_index_path}): {exc}')
                    skipped_incomplete += 1
                    continue

                ids: List[str] = stack_info.get('ids') or []
                k_value_map: Dict[str, Any] = stack_info.get('k_values', {})

                if not ids:
                    # Fallback: sort dict keys to keep a stable order.
                    def _sort_key(token: str) -> Tuple[int, int]:
                        try:
                            return (0, int(token))
                        except (TypeError, ValueError):
                            return (1, 0)

                    ids = sorted([str(key) for key in k_value_map.keys()], key=_sort_key)

                stack_paths: List[str] = []
                k_values: List[float] = []
                valid = True
                for stack_id in ids:
                    img_path = os.path.join(stack_dir, f'{stack_id}.png')
                    if not os.path.isfile(img_path):
                        print(f'[WARN] defocus-stack image missing: {img_path}')
                        valid = False
                        break
                    stack_paths.append(img_path)
                    try:
                        raw_val = k_value_map.get(str(stack_id), k_value_map.get(stack_id))
                        k_values.append(float(raw_val))
                    except (TypeError, ValueError):
                        try:
                            k_values.append(float(stack_id))
                        except (TypeError, ValueError):
                            valid = False
                            break

                if not valid or not stack_paths:
                    skipped_incomplete += 1
                    continue

                ref_path = stack_info.get('source_rgb')
                if not ref_path:
                    ref_path = stack_info.get('ref')
                if not ref_path:
                    print(f'[WARN] stack_index missing source_rgb: {stack_index_path}')
                    skipped_incomplete += 1
                    continue

                depth_ref = stack_info.get('source_depth')
                mask_ref = stack_info.get('source_mask')

                entry: Dict[str, Any] = {
                    'split': split_name,
                    'scene': scene_name,
                    'frame': sample_name,
                    'ref': ref_path,
                    'stack': stack_paths,
                    'k': k_values,
                    'stack_index': stack_index_path,
                    'size': stack_info.get('size'),
                }

                if depth_ref:
                    entry['depth'] = depth_ref
                if mask_ref:
                    entry['mask'] = mask_ref
                if 'source_size' in stack_info:
                    entry['source_size'] = stack_info['source_size']

                source_depth_ref = stack_info.get('source_depth')
                if source_depth_ref:
                    entry['source_depth'] = source_depth_ref
                for meta_key in [
                    'source_rgb',
                    'source_rgb_clean',
                    'source_rgb_final',
                    'source_mask',
                    'source_intrinsics',
                    'source_original_depth',
                    'sintel_input_variant',
                ]:
                    if stack_info.get(meta_key):
                        entry[meta_key] = stack_info[meta_key]

                manifest_entries.append(entry)

        if manifest_entries:
            manifest_path = os.path.join(split_dir, manifest_filename)
            ensure_dir(os.path.dirname(manifest_path))
            with open(manifest_path, 'w', encoding='utf-8') as f:
                for item in manifest_entries:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            results[split_name] = (manifest_path, len(manifest_entries))
            print(f'[INFO] NYUv2 {split_name} manifest written: {manifest_path} ({len(manifest_entries)} samples)')
            if skipped_incomplete:
                print(f'[INFO] NYUv2 {split_name} manifest skipped incomplete samples: {skipped_incomplete}')
        else:
            if skipped_incomplete:
                print(f'[WARN] NYUv2 {split_name} no complete samples (skipped {skipped_incomplete} directories)')

    return results


def _manifest_filename_for_dataset(dataset: str) -> str:
    mapping = {
        'vkitti2': 'manifest_bokeh_diffusion_vkitti2.jsonl',
        'kitti_test': 'manifest_bokeh_diffusion_kitti_test.jsonl',
        'kitti_train': 'manifest_bokeh_diffusion_kitti_train.jsonl',
        'eth3d': 'manifest_bokeh_diffusion_eth3d.jsonl',
        'hypersim': 'manifest_bokeh_diffusion_hypersim.jsonl',
        'make3d_kaggle': 'manifest_bokeh_diffusion_make3d_kaggle.jsonl',
        'make3d_official': 'manifest_bokeh_diffusion_make3d_official.jsonl',
        'opensun3d': 'manifest_bokeh_diffusion_opensun3d.jsonl',
        'middlebury2014': 'manifest_bokeh_diffusion_middlebury2014.jsonl',
        'ibims1': 'manifest_bokeh_diffusion_ibims1.jsonl',
        'hammer': 'manifest_bokeh_diffusion_hammer.jsonl',
        'bokeh_failure': 'manifest_bokeh_diffusion_bokeh_failure.jsonl',
        'sintel': 'manifest_bokeh_diffusion_sintel.jsonl',
        'sintel_final': 'manifest_bokeh_diffusion_sintel_final.jsonl',
    }
    return mapping.get(dataset, f'manifest_bokeh_diffusion_{dataset}.jsonl')


def _extract_metadata_from_relpath(dataset: str, rel_path: str) -> Dict[str, Any]:
    parts = rel_path.split(os.sep)
    info: Dict[str, Any] = {}

    if dataset == 'vkitti2':
        if len(parts) >= 1:
            info['scene_root'] = parts[0]
        if len(parts) >= 2:
            info['variation'] = parts[1]
        if len(parts) >= 3:
            info['camera'] = parts[-2]
            info['scene'] = os.path.join(*parts[:-1])
        if parts:
            info['frame'] = parts[-1]
    elif dataset in ('kitti_test', 'kitti_train'):
        if len(parts) >= 1:
            info['date'] = parts[0]
        if len(parts) >= 2:
            info['sequence'] = parts[1]
            info['scene'] = os.path.join(parts[0], parts[1]) if len(parts) >= 2 else parts[0]
        if parts:
            info['frame'] = parts[-1]
    elif dataset == 'hypersim':
        idx = 0
        if len(parts) >= 1 and not parts[0].startswith('ai_'):
            info['split'] = parts[0]
            idx = 1
        if len(parts) > idx:
            info['scene'] = parts[idx]
        if len(parts) > idx + 1:
            info['camera'] = parts[idx + 1]
        if len(parts) > idx + 2:
            info['frame'] = parts[idx + 2]
        elif parts:
            info['frame'] = parts[-1]
    elif dataset == 'eth3d':
        if len(parts) >= 1:
            info['scene'] = parts[0]
        if parts:
            info['frame'] = parts[-1]
    elif dataset == 'make3d_kaggle':
        if len(parts) >= 1:
            info['category'] = parts[0]
        if parts:
            info['frame'] = parts[-1]
    elif dataset == 'make3d_official':
        if len(parts) >= 1:
            info['category'] = parts[0]
        if parts:
            info['frame'] = parts[-1]
    elif dataset == 'opensun3d':
        if len(parts) >= 1:
            info['scene_id'] = parts[0]
        if len(parts) >= 2:
            info['sequence_id'] = parts[1]
        if parts:
            info['frame'] = parts[-1]
    elif dataset == 'middlebury2014':
        if parts:
            info['scene'] = parts[0]
            info['frame'] = parts[-1]
    elif dataset == 'ibims1':
        if parts:
            info['scene'] = parts[0]
            info['frame'] = parts[-1]
    elif dataset == 'hammer':
        if len(parts) >= 1:
            info['scene'] = parts[0]
        if len(parts) >= 2:
            info['sensor'] = parts[1]
        if parts:
            info['frame'] = parts[-1]
    elif dataset == 'bokeh_failure':
        info['rel_path'] = rel_path.replace(os.sep, '/')
    elif dataset in {'sintel', 'sintel_final'}:
        if len(parts) >= 1:
            info['scene'] = parts[0]
        if parts:
            info['frame'] = parts[-1]
    else:
        # Default: keep the relative path.
        info['rel_path'] = rel_path.replace(os.sep, '/')

    return info


def generate_generic_manifest(
    output_root: str,
    dataset: str,
    manifest_filename: Optional[str] = None,
) -> Dict[str, Tuple[str, int]]:
    results: Dict[str, Tuple[str, int]] = {}
    if not os.path.isdir(output_root):
        print(f'[WARN] cannot generate {dataset} manifest, output directory not found: {output_root}')
        return results

    manifest_filename = manifest_filename or _manifest_filename_for_dataset(dataset)

    entries: List[Dict[str, Any]] = []
    skipped_incomplete = 0

    for dirpath, _, filenames in os.walk(output_root):
        if 'stack_index.json' not in filenames:
            continue
        if os.path.basename(dirpath) != 'defocus_stack':
            continue

        stack_index_path = os.path.join(dirpath, 'stack_index.json')
        sample_dir = os.path.dirname(dirpath)

        try:
            with open(stack_index_path, 'r', encoding='utf-8') as f:
                stack_info = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f'[WARN] failed to parse stack_index ({stack_index_path}): {exc}')
            skipped_incomplete += 1
            continue

        ids: List[str] = stack_info.get('ids') or []
        k_value_map: Dict[str, Any] = stack_info.get('k_values', {})
        if not ids:
            if k_value_map:
                ids = sorted(k_value_map.keys(), key=lambda x: str(x))
            else:
                pngs = [
                    os.path.splitext(fname)[0]
                    for fname in sorted(os.listdir(dirpath))
                    if fname.lower().endswith('.png')
                ]
                ids = pngs

        stack_paths: List[str] = []
        k_values: List[float] = []
        valid = True
        for stack_id in ids:
            stack_img_path = os.path.join(dirpath, f'{stack_id}.png')
            if not os.path.isfile(stack_img_path):
                valid = False
                break
            stack_paths.append(stack_img_path)
            raw_val = k_value_map.get(str(stack_id))
            if raw_val is None:
                raw_val = k_value_map.get(stack_id)
            try:
                k_values.append(float(raw_val))
            except (TypeError, ValueError):
                try:
                    k_values.append(float(stack_id))
                except (TypeError, ValueError):
                    valid = False
                    break

        if not valid or not stack_paths:
            skipped_incomplete += 1
            continue

        ref_path = stack_info.get('source_rgb')
        if not ref_path:
            # Backward compatibility with the legacy output.
            legacy_ref = os.path.join(sample_dir, 'all_in_focus.png')
            ref_path = legacy_ref if os.path.isfile(legacy_ref) else None
        if not ref_path:
            print(f'[WARN] stack_index missing source_rgb: {stack_index_path}')
            skipped_incomplete += 1
            continue

        depth_ref = stack_info.get('source_depth')
        mask_ref = stack_info.get('source_mask')
        split_ref = stack_info.get('split')

        rel_path = os.path.relpath(sample_dir, output_root)
        entry: Dict[str, Any] = {
            'dataset': dataset,
            'rel_path': rel_path.replace(os.sep, '/'),
            'ref': ref_path,
            'stack': stack_paths,
            'k': k_values,
            'stack_index': stack_index_path,
        }

        if depth_ref:
            entry['depth'] = depth_ref
        if mask_ref:
            entry['mask'] = mask_ref
        if split_ref:
            entry['split'] = split_ref
        if 'size' in stack_info:
            entry['size'] = stack_info['size']
        for meta_key in [
            'source_rgb',
            'source_rgb_clean',
            'source_rgb_final',
            'source_depth',
            'source_mask',
            'source_intrinsics',
            'source_original_depth',
            'source_size',
            'sintel_input_variant',
        ]:
            if stack_info.get(meta_key):
                entry[meta_key] = stack_info[meta_key]

        entry.update({k: v for k, v in _extract_metadata_from_relpath(dataset, rel_path).items() if v is not None})

        entries.append(entry)

    if not entries:
        if skipped_incomplete:
            print(f'[WARN] {dataset} manifest has no complete samples (skipped {skipped_incomplete} directories)')
        else:
            print(f'[WARN] {dataset} manifest found no samples: {output_root}')
        return results

    entries.sort(key=lambda item: item.get('rel_path', ''))
    manifest_path = os.path.join(output_root, manifest_filename)
    ensure_dir(os.path.dirname(manifest_path))
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for item in entries:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f'[INFO] {dataset} manifest written: {manifest_path} ({len(entries)} samples)')
    if skipped_incomplete:
        print(f'[INFO] {dataset} manifest skipped incomplete samples: {skipped_incomplete}')

    results[dataset] = (manifest_path, len(entries))
    return results


def resize_preserve_aspect(arr: np.ndarray, target_short: int, interpolation: int) -> np.ndarray:
    """Resize the short side to target_short while preserving the aspect ratio."""
    if target_short is None or target_short <= 0:
        return arr
    height, width = arr.shape[:2]
    short_side = min(height, width)
    if short_side == 0:
        return arr
    if short_side == target_short:
        return arr
    scale = float(target_short) / float(short_side)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    return cv2.resize(arr, (new_width, new_height), interpolation=interpolation)


def _inject_runtime_conditions_to_adapter(
    bokeh_adapter: BokehFluxControlAdapter,
    camera_ann_tensor: torch.Tensor,
    perform_swap: bool,
    is_i2i: bool,
    batch_swap_ids: Optional[torch.Tensor] = None,
):
    """Inject the K condition and the grounded-attention flag into the adapter's attention processors (reused from the I2I code)."""
    try:
        with torch.no_grad():
            # Handle the DistributedDataParallel wrapper.
            adapter_model = bokeh_adapter.module if hasattr(bokeh_adapter, 'module') else bokeh_adapter
            camera_embeds = adapter_model.embedding_layer(camera_ann_tensor)
        num_modules = 0
        for module in adapter_model.adapter_modules:
            try:
                module.stored_camera_embeds = camera_embeds
                module.stored_perform_swap = perform_swap
                module.stored_batch_swap_ids = batch_swap_ids
                module.stored_is_i2i = is_i2i
                module.use_stored_embeds = True
                num_modules += 1
            except AttributeError:
                pass
    except Exception as e:
        print(f"Failed to inject adapter conditions: {e}")


def generate_i2i_bokeh_image(
    kontext_pipeline,
    bokeh_adapter,
    input_image: Image.Image,
    prompt,  # str or list[str]
    dof_cond=None,  # single K value, ignored when k_values_batch is used
    k_values_batch: Optional[List[float]] = None,  # batch of multiple K values
    K_max: float = 30.0,
    guidance_scale: float = 1.0,
    num_inference_steps: int = 20,
    device: torch.device = None,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
    vae_scale_factor: int = 8,
    assigned_size: Optional[int] = None
):
    """Generate a bokeh image with the I2I model (adapted from eval_i2i_syn_foreground.py, with batch support)."""

    # Prepare camera condition (aligned with inference_flux_v4.py).
    if k_values_batch is not None and len(k_values_batch) >= 2:
        # Batch mode: build camera condition per K value, applying clamp.
        k_norms = [max(1.0, min(30.0, float(k))) / 30.0 for k in k_values_batch]
        camera_ann_normalized = torch.tensor([[float(kn)] for kn in k_norms], device=device, dtype=dtype)
        batch_size = len(k_values_batch)
        print(f"Batch mode: generating bokeh images for {batch_size} K values: {k_values_batch}")
        print(f"Camera condition (batch): K_norm={k_norms}")
    else:
        # Single-K mode.
        if dof_cond is None:
            raise ValueError("Either dof_cond or k_values_batch must be provided")
        k_norm_single = max(1.0, min(30.0, float(dof_cond))) / 30.0
        camera_ann_normalized = torch.tensor([[float(k_norm_single)]], device=device, dtype=dtype)
        batch_size = 1
        print(f"Camera condition: K_norm={k_norm_single:.4f}")

    # Print mode info (aligned with inference_flux_v4.py).
    print(f"Mode: I2I, batch_size={batch_size}")
    print(f"I2I mode - Grounded Attention: disabled")

    # Inject adapter conditions (one shot for all K values).
    _inject_runtime_conditions_to_adapter(
        bokeh_adapter=bokeh_adapter,
        camera_ann_tensor=camera_ann_normalized,
        perform_swap=False,
        is_i2i=True,
        batch_swap_ids=None,
    )

    # Handle prompt: support a list of prompts in batch mode.
    if isinstance(prompt, list):
        # Batch mode: prompt count must match K count.
        if len(prompt) != batch_size:
            raise ValueError(f"prompt list length ({len(prompt)}) must match K count ({batch_size})")
        prompt_final = prompt
    else:
        # Single-prompt mode: duplicate the prompt across the batch.
        prompt_final = [prompt] * batch_size if batch_size > 1 else prompt

    # Decide processing based on whether a fixed size is specified.
    if assigned_size is not None:
        new_height = assigned_size
        new_width = assigned_size
        orig_width, orig_height = input_image.size
        if (new_width != orig_width) or (new_height != orig_height):
            input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
    else:
        # Use a VAE-aligned adaptive size.
        orig_width, orig_height = input_image.size
        multiple = int(vae_scale_factor) * 2  # Kontext's 2x2 packing

        new_height = (orig_height // multiple) * multiple
        new_width = (orig_width // multiple) * multiple

        if (new_width != orig_width) or (new_height != orig_height):
            input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

    # Set up the generator (aligned with inference_flux_v4.py).
    if batch_size >= 2:
        # Batch generation: same seed for every sample's generator.
        generator = [torch.Generator(device="cpu").manual_seed(seed) for _ in range(batch_size)]
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    # Run I2I generation via the Kontext pipeline (fully aligned with inference_flux_v4.py).
    with torch.no_grad():
        try:
            result = kontext_pipeline(
                prompt=prompt_final,
                image=input_image,
                height=new_height,
                width=new_width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                max_area=new_height * new_width,
                true_cfg_scale=1.0,  # mirrors the true_cfg arg in inference_flux_v4.py
                negative_prompt=None,
                generator=generator,
                _auto_resize=False,
            )

            # Batch mode returns a list of images; single mode returns one image (matches inference_flux_v4.py).
            if batch_size >= 2:
                generated_image = result.images  # list of images
                print(f"Batch generation done: produced {batch_size} bokeh images for the K values")
            else:
                generated_image = result.images[0]  # single image

        except Exception as e:
            print(f"I2I generation failed: {e}")
            import traceback
            traceback.print_exc()
            return [input_image] * batch_size if batch_size > 1 else input_image

    # Reset the processor state.
    try:
        # Handle the DistributedDataParallel wrapper.
        adapter_model = bokeh_adapter.module if hasattr(bokeh_adapter, 'module') else bokeh_adapter
        for module in adapter_model.adapter_modules:
            if hasattr(module, "use_stored_embeds"):
                module.use_stored_embeds = False
            if hasattr(module, "stored_camera_embeds"):
                module.stored_camera_embeds = None
            if hasattr(module, "stored_perform_swap"):
                module.stored_perform_swap = False
            if hasattr(module, "stored_batch_swap_ids"):
                module.stored_batch_swap_ids = None
            if hasattr(module, "stored_is_i2i"):
                module.stored_is_i2i = False
    except Exception:
        pass

    return generated_image


def main():
    args = parse_args()

    # Seed RNG.
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    ensure_dir(args.output_root)

    # Create accelerator.
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    # Determine dtype.
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load model components.
    if accelerator.is_main_process:
        print("Loading FluxKontextPipeline...")
    kontext_pipeline = FluxKontextPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype
    )

    transformer = kontext_pipeline.transformer
    vae = kontext_pipeline.vae

    # Freeze parameters.
    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    # Fetch VAE scale factor.
    vae_scale_factor = getattr(kontext_pipeline, "vae_scale_factor", getattr(vae, "scale_factor", 8))

    kontext_pipeline = kontext_pipeline.to(accelerator.device, dtype=weight_dtype)

    # Load adapter.
    if accelerator.is_main_process:
        print("Loading BokehFluxControlAdapter...")
    bokeh_adapter = BokehFluxControlAdapter(
        transformer,
        blocks=args.blocks,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        unfreeze_q=args.unfreeze_q,
        unfreeze_k=args.unfreeze_k,
        ckpt_path=args.adapter_ckpt,
    ).to(accelerator.device, dtype=weight_dtype)

    # Prepare model.
    bokeh_adapter = accelerator.prepare(bokeh_adapter)
    bokeh_adapter.eval()

    if accelerator.is_main_process:
        print(f"Dataset mode: {args.dataset}")

    if args.dataset == 'vkitti2':
        items = list_vkitti2_images(args.rgb_root, allowed_cameras=args.vkitti2_cameras)
        if accelerator.is_main_process:
            camera_info = 'all cameras' if args.vkitti2_cameras is None else ', '.join(sorted(args.vkitti2_cameras))
            print(f'Total images found (Virtual KITTI 2, {camera_info}): {len(items)}')
    elif args.dataset == 'hypersim':
        items = list_hypersim_images(args.hypersim_root, args.hypersim_split_csv, args.hypersim_split)
        if accelerator.is_main_process:
            if args.hypersim_split.lower() in {'all', '*'}:
                split_counter = Counter(split for split, *_ in items)
                print(f'Total pairs found (Hypersim all splits): {len(items)}')
                for split_name in sorted(split_counter.keys()):
                    print(f'  - {split_name}: {split_counter[split_name]}')
            else:
                print(f'Total pairs found (Hypersim {args.hypersim_split} split): {len(items)}')
    elif args.dataset == 'nyuv2':
        items = list_nyuv2_images(args.nyuv2_root)
        if accelerator.is_main_process:
            print(f'Total pairs found (NYUv2 official train/test): {len(items)}')
    elif args.dataset == 'eth3d':
        items = list_eth3d_images(args.eth3d_root)
        if accelerator.is_main_process:
            print(f'Total pairs found (ETH3D high-res MVS): {len(items)}')
    elif args.dataset == 'make3d_kaggle':
        items = list_make3d_images(args.make3d_kaggle_root)
        if accelerator.is_main_process:
            print(f'Total pairs found (Make3D Kaggle dataset): {len(items)}')
    elif args.dataset == 'make3d_official':
        items = list_make3d_official_images(args.make3d_official_root)
        if accelerator.is_main_process:
            print(f'Total pairs found (Make3D Official Test134 dataset): {len(items)}')
    elif args.dataset == 'opensun3d':
        items = list_opensun3d_frames(args.opensun3d_root)
        if accelerator.is_main_process:
            print(f'Total frames found (OpenSUN3D lowres_wide): {len(items)}')
    elif args.dataset == 'middlebury2014':
        items = list_middlebury2014_images(args.middlebury_root)
        if accelerator.is_main_process:
            print(f'Total samples found (Middlebury Stereo 2014 Perfect): {len(items)}')
    elif args.dataset == 'ibims1':
        items = list_ibims1_images(args.ibims1_root)
        if accelerator.is_main_process:
            print(f'Total samples found (iBims-1 eval): {len(items)}')
            if items:
                prefix_counter = Counter(sample.split('_')[0] for sample, *_ in items)
                top_categories = ', '.join(f'{k}:{prefix_counter[k]}' for k in sorted(prefix_counter.keys()))
                print(f'Category distribution: {top_categories}')
    elif args.dataset == 'hammer':
        items = list_hammer_images(args.hammer_root, args.hammer_depth_type)
        if accelerator.is_main_process:
            scene_counter = Counter(item[0] for item in items)
            print(f'Total frames found (HAMMER {args.hammer_depth_type}): {len(items)}')
            for scene_name in sorted(scene_counter.keys()):
                print(f'  - {scene_name}: {scene_counter[scene_name]}')
    elif args.dataset in {'sintel', 'sintel_final'}:
        items = list_sintel_images(args.sintel_root)
        if accelerator.is_main_process:
            variant = 'final' if args.dataset == 'sintel_final' else 'clean'
            print(f'Total frames found (Sintel Depth training, input={variant}): {len(items)}')
    elif args.dataset == 'bokeh_failure':
        items = load_bokeh_failure_manifest(args.bokeh_failure_manifest, args.bokeh_failure_root)
        if accelerator.is_main_process:
            print(f'Total samples found (bokeh_failure manifest): {len(items)}')
    elif args.dataset == 'kitti_test':
        items = get_kitti_items_from_test_files(args.kitti_data_root, args.kitti_test_files)
        if accelerator.is_main_process:
            print(f'Total samples found (KITTI list): {len(items)}')
    elif args.dataset == 'kitti_train':
        items = get_kitti_items_from_test_files(args.kitti_data_root, args.kitti_train_files)
        # Monocular training keeps left-camera frames only, matching the common monodepth setup.
        original_count = len(items)
        items = [item for item in items if item[2] == 'l']
        if accelerator.is_main_process:
            print(f'Total samples found (KITTI train list): {len(items)}')
            depth_root_status = "found" if os.path.isdir(args.kitti_train_depths_root) else "missing"
            print(f'KITTI train depth root: {args.kitti_train_depths_root} ({depth_root_status})')
            if len(items) != original_count:
                print(f'Filtered KITTI train items to left camera only: {len(items)} / {original_count}')
    else:
        if accelerator.is_main_process:
            print(f'[ERROR] unsupported dataset: {args.dataset}')
        return

    if args.max_samples is not None:
        items = items[:args.max_samples]
        if accelerator.is_main_process:
            print(f'Limited to first {args.max_samples} samples')

    # When inverse mode is enabled, walk the dataset in reverse order (last to first).
    if args.inverse:
        items = items[::-1]
        if accelerator.is_main_process:
            print(f'inverse mode enabled: dataset traversal order reversed (last to first)')

    # Distributed split: use strided slicing for better load balancing, avoiding long barrier waits
    # when some processes finish early.
    # e.g. with 4 processes: rank0 handles 0,4,8,...; rank1 handles 1,5,9,...
    process_items = items[accelerator.process_index::max(1, accelerator.num_processes)]

    if accelerator.is_main_process:
        print(f"Each process handles {len(process_items)} samples")

    gt_depths_array: Optional[np.ndarray] = None
    if args.dataset == 'kitti_test' and args.kitti_gt_depths and os.path.exists(args.kitti_gt_depths):
        try:
            gt_depths_array = load_kitti_gt_depths(args.kitti_gt_depths)
            if accelerator.is_main_process:
                print(f'Loaded KITTI depth ground truth: {args.kitti_gt_depths} ({len(gt_depths_array)} frames)')
        except Exception as exc:
            if accelerator.is_main_process:
                print(f'[WARN] failed to load KITTI depth ground truth {args.kitti_gt_depths}: {exc}')
    elif args.dataset == 'kitti_test' and accelerator.is_main_process:
        print('[WARN] no valid KITTI depth ground truth provided, skipping depth.npy save')

    kitti_train_depth_root: Optional[str] = None
    if args.dataset == 'kitti_train':
        kitti_train_depth_root = args.kitti_train_depths_root or None
        if accelerator.is_main_process and kitti_train_depth_root and not os.path.isdir(kitti_train_depth_root):
            print(f'[WARN] provided KITTI training depth directory not found: {kitti_train_depth_root}')

    # K-value sequence (ensure the upper bound is included).
    k_vals = []
    v = float(args.k_min)
    while v <= float(args.k_max):
        k_vals.append(float(v))
        v += float(args.k_step)
    if abs(k_vals[-1] - float(args.k_max)) > 1e-6:
        k_vals.append(float(args.k_max))

    if accelerator.is_main_process:
        print(f'K sequence: {k_vals}')

    # Default prompt template (see inference_flux_v4.py).
    dof_cond_default_tpl = (
        "Set dof_cond = {value:.2f} (stronger background defocus); "
        "preserve subject sharpness; keep composition, lighting, and colors unchanged."
    )
    prompt_template = args.prompt_template if args.prompt_template is not None else dof_cond_default_tpl

    if accelerator.is_main_process:
        print(f'Using prompt template: {prompt_template}')

    # Synchronize starting point (once is enough; avoid long waits when per-process progress diverges).
    accelerator.wait_for_everyone()

    resize_enabled = bool(args.resize)
    resize_target = max(1, int(args.size)) if args.size is not None else 518

    # Count skipped and processed samples.
    skipped_count = 0
    processed_count = 0

    for sample_idx, item in enumerate(process_items):
        try:
            source_mask_ref: Optional[str] = None
            source_intrinsics_ref: Optional[str] = None
            extra_meta: Dict[str, Any] = {}
            if args.dataset == 'vkitti2':
                scene_rel, stem, rgb_path = item
                depth_path = find_matching_depth(args.depth_root, scene_rel, stem)
                if depth_path is None:
                    if accelerator.is_main_process:
                        print(f'[WARN] Depth not found for {scene_rel}/{stem}, skip')
                    continue
                img = Image.open(rgb_path).convert('RGB')
                img_np = np.array(img).astype(np.float32) / 255.0
                if args.assigned_size is not None:
                    img_np = center_crop_to_size(img_np, args.assigned_size, args.assigned_size)

                depth_np = load_depth_any(depth_path)
                if depth_np.ndim == 3:
                    depth_np = depth_np[..., 0]
                if args.assigned_size is not None:
                    depth_np = center_crop_to_size(depth_np, args.assigned_size, args.assigned_size)

                sample_dir = os.path.join(args.output_root, scene_rel, stem)

                # If the sample has already been fully generated, skip it.
                if check_sample_complete(sample_dir, len(k_vals)):
                    skipped_count += 1
                    if accelerator.is_main_process and sample_idx % 50 == 0:
                        print(f'[SKIP] process {accelerator.process_index} skipping completed sample {sample_idx + 1}/{len(process_items)}: {scene_rel}/{stem}')
                    continue

                stack_dir = os.path.join(sample_dir, 'defocus_stack')
                ensure_dir(stack_dir)

                source_rgb = rgb_path
                source_depth_ref = depth_path
                sample_desc = f'{scene_rel}/{stem}'
            elif args.dataset in {'sintel', 'sintel_final'}:
                scene, stem, clean_path, depth_path, cam_path, final_path = item
                input_path = final_path if args.dataset == 'sintel_final' else clean_path
                try:
                    img = Image.open(input_path).convert('RGB')
                    img_np = np.array(img).astype(np.float32) / 255.0
                except Exception as exc:
                    if accelerator.is_main_process:
                        target = 'final' if args.dataset == 'sintel_final' else 'clean'
                        print(f'[WARN] failed to read Sintel {target} image {scene}/{stem}: {exc}')
                    continue

                try:
                    depth_np = load_sintel_depth(depth_path)
                except Exception as exc:
                    if accelerator.is_main_process:
                        print(f'[WARN] failed to read Sintel depth {scene}/{stem}: {exc}')
                    continue

                if args.assigned_size is not None:
                    img_np = center_crop_to_size(img_np, args.assigned_size, args.assigned_size)
                    depth_np = center_crop_to_size(depth_np, args.assigned_size, args.assigned_size)

                sample_dir = os.path.join(args.output_root, scene, stem)

                if check_sample_complete(sample_dir, len(k_vals)):
                    skipped_count += 1
                    if accelerator.is_main_process and sample_idx % 50 == 0:
                        print(f'[SKIP] process {accelerator.process_index} skipping completed sample {sample_idx + 1}/{len(process_items)}: {scene}/{stem}')
                    continue

                stack_dir = os.path.join(sample_dir, 'defocus_stack')
                ensure_dir(stack_dir)

                source_rgb = input_path
                source_depth_ref = depth_path
                sample_desc = f'sintel/{scene}/{stem}'
                source_intrinsics_ref = cam_path if cam_path else None
                extra_meta['source_rgb_clean'] = clean_path
                if final_path:
                    extra_meta['source_rgb_final'] = final_path
                extra_meta['sintel_input_variant'] = 'final' if args.dataset == 'sintel_final' else 'clean'
            elif args.dataset == 'hypersim':
                split_name, scene, camera, frame_id, color_path, depth_path = item
                try:
                    img_np = load_hypersim_color(color_path)
                except Exception as exc:
                    if accelerator.is_main_process:
                        print(f'[WARN] failed to read Hypersim image {split_name}/{scene}/{camera}/{frame_id}: {exc}')
                    continue

                try:
                    depth_np = load_hypersim_depth(depth_path)
                except Exception as exc:
                    if accelerator.is_main_process:
                        print(f'[WARN] failed to read Hypersim depth {split_name}/{scene}/{camera}/{frame_id}: {exc}')
                    continue

                if args.assigned_size is not None:
                    img_np = center_crop_to_size(img_np, args.assigned_size, args.assigned_size)
                    depth_np = center_crop_to_size(depth_np, args.assigned_size, args.assigned_size)

                frame_dir = _format_hypersim_frame_dir(frame_id)
                split_dir = split_name or 'unspecified'
                output_root = args.output_root
                normalized_split_dir = split_dir.replace(os.sep, '_')
                if os.path.basename(os.path.normpath(output_root)) == normalized_split_dir:
                    sample_dir = os.path.join(output_root, scene, camera, frame_dir)
                else:
                    sample_dir = os.path.join(output_root, normalized_split_dir, scene, camera, frame_dir)

                # If the sample has already been fully generated, skip it.
                if check_sample_complete(sample_dir, len(k_vals)):
                    skipped_count += 1
                    if accelerator.is_main_process and sample_idx % 50 == 0:
                        print(f'[SKIP] process {accelerator.process_index} skipping completed sample {sample_idx + 1}/{len(process_items)}: {split_dir}/{scene}/{camera}/{frame_dir}')
                    continue

                stack_dir = os.path.join(sample_dir, 'defocus_stack')
                ensure_dir(stack_dir)

                # Hypersim inputs stay in HDF5 format; only record the source path in metadata, do not copy.
                source_rgb = color_path
                source_depth_ref = depth_path
                sample_desc = f'hypersim/{split_dir}/{scene}/{camera}/{frame_dir}'
                source_mask_ref = None
                extra_meta['split'] = split_dir
            elif args.dataset == 'nyuv2':
                rel_dir, stem, rgb_path, depth_path = item
                if depth_path is None or not os.path.exists(depth_path):
                    if accelerator.is_main_process:
                        print(f'[WARN] NYUv2 depth missing for {rel_dir}/{stem}, skip')
                    continue

                img = Image.open(rgb_path).convert('RGB')
                img_np = np.array(img).astype(np.float32) / 255.0
                if args.assigned_size is not None:
                    img_np = center_crop_to_size(img_np, args.assigned_size, args.assigned_size)

                depth_np = load_depth_any(depth_path)
                if depth_np.ndim == 3:
                    depth_np = depth_np[..., 0]
                if args.assigned_size is not None:
                    depth_np = center_crop_to_size(depth_np, args.assigned_size, args.assigned_size)

                sample_dir = os.path.join(args.output_root, rel_dir, stem)

                # If the sample has already been fully generated, skip it.
                if check_sample_complete(sample_dir, len(k_vals)):
                    skipped_count += 1
                    if accelerator.is_main_process and sample_idx % 50 == 0:
                        print(f'[SKIP] process {accelerator.process_index} skipping completed sample {sample_idx + 1}/{len(process_items)}: nyuv2/{rel_dir}/{stem}')
                    continue

                stack_dir = os.path.join(sample_dir, 'defocus_stack')
                ensure_dir(stack_dir)

                source_rgb = rgb_path
                source_depth_ref = depth_path
                sample_desc = f'nyuv2/{rel_dir}/{stem}'
                source_mask_ref = None
            elif args.dataset == 'eth3d':
                scene, stem, rgb_path, depth_path, mask_path = item
                if depth_path is None or not os.path.exists(depth_path):
                    if accelerator.is_main_process:
                        print(f'[WARN] ETH3D depth missing for {scene}/{stem}, skip')
                    continue

                img = Image.open(rgb_path).convert('RGB')
                img_np = np.array(img).astype(np.float32) / 255.0
                if args.assigned_size is not None:
                    img_np = center_crop_to_size(img_np, args.assigned_size, args.assigned_size)

                try:
                    depth_np_full = load_eth3d_depth(depth_path, img.size)
                except Exception as exc:
                    if accelerator.is_main_process:
                        print(f'[WARN] failed to read ETH3D depth {scene}/{stem}: {exc}')
                    continue

                if args.assigned_size is not None:
                    depth_np = center_crop_to_size(depth_np_full, args.assigned_size, args.assigned_size)
                else:
                    depth_np = depth_np_full
                del depth_np_full

                sample_dir = os.path.join(args.output_root, scene, stem)

                # If the sample has already been fully generated, skip it.
                if check_sample_complete(sample_dir, len(k_vals)):
                    skipped_count += 1
                    if accelerator.is_main_process and sample_idx % 50 == 0:
                        print(f'[SKIP] process {accelerator.process_index} skipping completed sample {sample_idx + 1}/{len(process_items)}: eth3d/{scene}/{stem}')
                    continue

                stack_dir = os.path.join(sample_dir, 'defocus_stack')
                ensure_dir(stack_dir)

                source_rgb = rgb_path
                source_depth_ref = depth_path
                source_mask_ref = mask_path if mask_path and os.path.exists(mask_path) else None
                sample_desc = f'eth3d/{scene}/{stem}'
            elif args.dataset == 'make3d_kaggle':
                category, stem, rgb_path, depth_path = item
                if depth_path is None or not os.path.exists(depth_path):
                    if accelerator.is_main_process:
                        print(f'[WARN] Make3D Kaggle depth missing for {category}/{stem}, skip')
                    continue

                img = Image.open(rgb_path).convert('RGB')
                img_np = np.array(img).astype(np.float32) / 255.0
                if args.assigned_size is not None:
                    img_np = center_crop_to_size(img_np, args.assigned_size, args.assigned_size)

                try:
                    depth_lowres = load_make3d_depth(depth_path)
                except Exception as exc:
                    if accelerator.is_main_process:
                        print(f'[WARN] failed to read Make3D Kaggle depth {category}/{stem}: {exc}')
                    continue

                depth_np = cv2.resize(
                    depth_lowres,
                    (img_np.shape[1], img_np.shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
                if args.assigned_size is not None:
                    depth_np = center_crop_to_size(depth_np, args.assigned_size, args.assigned_size)
                depth_np = depth_np.astype(np.float32)

                sample_dir = os.path.join(args.output_root, category, stem)

                # If the sample has already been fully generated, skip it.
                if check_sample_complete(sample_dir, len(k_vals)):
                    skipped_count += 1
                    if accelerator.is_main_process and sample_idx % 50 == 0:
                        print(f'[SKIP] process {accelerator.process_index} skipping completed sample {sample_idx + 1}/{len(process_items)}: make3d_kaggle/{category}/{stem}')
                    continue

                stack_dir = os.path.join(sample_dir, 'defocus_stack')
                ensure_dir(stack_dir)

                source_rgb = rgb_path
                source_depth_ref = depth_path
                source_mask_ref = None
                sample_desc = f'make3d_kaggle/{category}/{stem}'
            elif args.dataset == 'make3d_official':
                category, stem, rgb_path, depth_path = item
                if depth_path is None or not os.path.exists(depth_path):
                    if accelerator.is_main_process:
                        print(f'[WARN] Make3D Official depth missing for {category}/{stem}, skip')
                    continue

                img = Image.open(rgb_path).convert('RGB')
                img_np = np.array(img).astype(np.float32) / 255.0
                if args.assigned_size is not None:
                    img_np = center_crop_to_size(img_np, args.assigned_size, args.assigned_size)

                try:
                    depth_lowres = load_make3d_official_depth(depth_path)
                except Exception as exc:
                    if accelerator.is_main_process:
                        print(f'[WARN] failed to read Make3D Official depth {category}/{stem}: {exc}')
                    continue

                depth_np = cv2.resize(
                    depth_lowres,
                    (img_np.shape[1], img_np.shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
                if args.assigned_size is not None:
                    depth_np = center_crop_to_size(depth_np, args.assigned_size, args.assigned_size)
                depth_np = depth_np.astype(np.float32)

                sample_dir = os.path.join(args.output_root, category, stem)

                # If the sample has already been fully generated, skip it.
                if check_sample_complete(sample_dir, len(k_vals)):
                    skipped_count += 1
                    if accelerator.is_main_process and sample_idx % 50 == 0:
                        print(f'[SKIP] process {accelerator.process_index} skipping completed sample {sample_idx + 1}/{len(process_items)}: make3d_official/{category}/{stem}')
                    continue

                stack_dir = os.path.join(sample_dir, 'defocus_stack')
                ensure_dir(stack_dir)

                source_rgb = rgb_path
                source_depth_ref = depth_path
                source_mask_ref = None
                sample_desc = f'make3d_official/{category}/{stem}'
            elif args.dataset == 'opensun3d':
                scene_id, sequence_id, frame_stem, rgb_path, depth_path, intr_path = item
                if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                    if accelerator.is_main_process:
                        print(f'[WARN] OpenSUN3D missing assets for {scene_id}/{sequence_id}/{frame_stem}, skip')
                    continue

                img = Image.open(rgb_path).convert('RGB')
                img_np = np.array(img).astype(np.float32) / 255.0
                if args.assigned_size is not None:
                    img_np = center_crop_to_size(img_np, args.assigned_size, args.assigned_size)

                depth_np = load_depth_any(depth_path)
                if depth_np.ndim == 3:
                    depth_np = depth_np[..., 0]
                if args.assigned_size is not None:
                    depth_np = center_crop_to_size(depth_np, args.assigned_size, args.assigned_size)
                depth_np = depth_np.astype(np.float32) / 1000.0  # mm -> m

                sample_dir = os.path.join(args.output_root, str(scene_id), str(sequence_id), frame_stem)

                # If the sample has already been fully generated, skip it.
                if check_sample_complete(sample_dir, len(k_vals)):
                    skipped_count += 1
                    if accelerator.is_main_process and sample_idx % 50 == 0:
                        print(f'[SKIP] process {accelerator.process_index} skipping completed sample {sample_idx + 1}/{len(process_items)}: opensun3d/{scene_id}/{sequence_id}/{frame_stem}')
                    continue

                stack_dir = os.path.join(sample_dir, 'defocus_stack')
                ensure_dir(stack_dir)

                source_rgb = rgb_path
                source_depth_ref = depth_path
                source_intrinsics_ref = intr_path
                sample_desc = f'opensun3d/{scene_id}/{sequence_id}/{frame_stem}'
            elif args.dataset == 'middlebury2014':
                scene_name, rgb_path, disp_path, calib_path, baseline_m, focal_px, disp_samples_path, disp_sd_path = item
                if not os.path.exists(rgb_path) or not os.path.exists(disp_path):
                    if accelerator.is_main_process:
                        print(f'[WARN] Middlebury2014 missing assets for {scene_name}, skip')
                    continue

                img = Image.open(rgb_path).convert('RGB')
                img_np = np.array(img).astype(np.float32) / 255.0
                if args.assigned_size is not None:
                    img_np = center_crop_to_size(img_np, args.assigned_size, args.assigned_size)

                try:
                    depth_np = load_middlebury_depth(disp_path, baseline_m, focal_px)
                except Exception as exc:
                    if accelerator.is_main_process:
                        print(f'[WARN] failed to read Middlebury2014 disparity {scene_name}: {exc}')
                    continue

                if args.assigned_size is not None:
                    depth_np = center_crop_to_size(depth_np, args.assigned_size, args.assigned_size)

                sample_dir = os.path.join(args.output_root, scene_name)

                if check_sample_complete(sample_dir, len(k_vals)):
                    skipped_count += 1
                    if accelerator.is_main_process and sample_idx % 50 == 0:
                        print(f'[SKIP] process {accelerator.process_index} skipping completed sample {sample_idx + 1}/{len(process_items)}: middlebury2014/{scene_name}')
                    continue

                stack_dir = os.path.join(sample_dir, 'defocus_stack')
                ensure_dir(stack_dir)

                source_rgb = rgb_path
                source_depth_ref = None
                source_mask_ref = None
                source_intrinsics_ref = calib_path if calib_path and os.path.exists(calib_path) else None
                sample_desc = f'middlebury2014/{scene_name}'
                extra_meta['scene'] = scene_name
                if os.path.exists(disp_path):
                    extra_meta['source_disparity'] = disp_path
                if calib_path and os.path.exists(calib_path):
                    extra_meta['source_calibration'] = calib_path
                extra_meta['source_baseline_m'] = float(baseline_m)
                extra_meta['source_focal_length_px'] = float(focal_px)
                if disp_samples_path and os.path.exists(disp_samples_path):
                    extra_meta['source_disp_samples'] = disp_samples_path
                if disp_sd_path and os.path.exists(disp_sd_path):
                    extra_meta['source_disp_uncertainty'] = disp_sd_path
            elif args.dataset == 'ibims1':
                scene_name, rgb_path, depth_path, mask_invalid_path, mask_transp_path, calib_path = item
                if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                    if accelerator.is_main_process:
                        print(f'[WARN] iBims-1 missing assets for {scene_name}, skip')
                    continue

                img = Image.open(rgb_path).convert('RGB')
                img_np = np.array(img).astype(np.float32) / 255.0
                if args.assigned_size is not None:
                    img_np = center_crop_to_size(img_np, args.assigned_size, args.assigned_size)

                try:
                    depth_np = load_ibims1_depth(depth_path)
                except Exception as exc:
                    if accelerator.is_main_process:
                        print(f'[WARN] failed to read iBims-1 depth {scene_name}: {exc}')
                    continue

                mask_combined: Optional[np.ndarray] = None
                if mask_invalid_path and os.path.exists(mask_invalid_path):
                    mask_invalid = np.array(Image.open(mask_invalid_path), dtype=np.uint8)
                    mask_combined = mask_invalid > 0
                if mask_transp_path and os.path.exists(mask_transp_path):
                    mask_transp = np.array(Image.open(mask_transp_path), dtype=np.uint8)
                    mask_transp_bool = mask_transp > 0
                    mask_combined = mask_transp_bool if mask_combined is None else (mask_combined & mask_transp_bool)
                    extra_meta['mask_transparent'] = mask_transp_path

                if mask_combined is not None:
                    depth_np = depth_np * mask_combined.astype(np.float32)

                if args.assigned_size is not None:
                    depth_np = center_crop_to_size(depth_np, args.assigned_size, args.assigned_size)

                sample_dir = os.path.join(args.output_root, scene_name)

                if check_sample_complete(sample_dir, len(k_vals)):
                    skipped_count += 1
                    if accelerator.is_main_process and sample_idx % 50 == 0:
                        print(f'[SKIP] process {accelerator.process_index} skipping completed sample {sample_idx + 1}/{len(process_items)}: ibims1/{scene_name}')
                    continue

                stack_dir = os.path.join(sample_dir, 'defocus_stack')
                ensure_dir(stack_dir)

                source_rgb = rgb_path
                source_depth_ref = depth_path
                source_mask_ref = mask_invalid_path if mask_invalid_path and os.path.exists(mask_invalid_path) else None
                if calib_path and os.path.exists(calib_path):
                    source_intrinsics_ref = calib_path
                sample_desc = f'ibims1/{scene_name}'
                extra_meta['scene'] = scene_name
            elif args.dataset == 'hammer':
                scene_name, sensor_key, depth_dir_name, stem, rgb_path, depth_path, intr_path = item
                if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                    if accelerator.is_main_process:
                        print(f'[WARN] HAMMER missing assets: {scene_name}/{sensor_key}/{depth_dir_name}/{stem}, skip')
                    continue

                img = Image.open(rgb_path).convert('RGB')
                img_np = np.array(img).astype(np.float32) / 255.0
                if args.assigned_size is not None:
                    img_np = center_crop_to_size(img_np, args.assigned_size, args.assigned_size)

                depth_np = load_depth_any(depth_path)
                if depth_np.ndim == 3:
                    depth_np = depth_np[..., 0]
                if args.assigned_size is not None:
                    depth_np = center_crop_to_size(depth_np, args.assigned_size, args.assigned_size)
                depth_np = depth_np.astype(np.float32) / 1000.0  # mm -> m

                sample_dir = os.path.join(args.output_root, scene_name, sensor_key, stem)

                if check_sample_complete(sample_dir, len(k_vals)):
                    skipped_count += 1
                    if accelerator.is_main_process and sample_idx % 50 == 0:
                        print(f'[SKIP] process {accelerator.process_index} skipping completed sample {sample_idx + 1}/{len(process_items)}: hammer/{scene_name}/{sensor_key}/{stem}')
                    continue

                stack_dir = os.path.join(sample_dir, 'defocus_stack')
                ensure_dir(stack_dir)

                source_rgb = rgb_path
                source_depth_ref = depth_path
                source_intrinsics_ref = intr_path
                sample_desc = f'hammer/{scene_name}/{sensor_key}/{stem}'
                extra_meta['scene'] = scene_name
                extra_meta['sensor'] = sensor_key
                extra_meta['depth_modality'] = depth_dir_name
            elif args.dataset == 'bokeh_failure':
                sample_info = item
                all_in_focus_path = sample_info['all_in_focus_path']
                depth_path = sample_info['depth_path']

                if not os.path.exists(all_in_focus_path):
                    if accelerator.is_main_process:
                        print(f'[WARN] bokeh_failure all_in_focus not found: {all_in_focus_path}')
                    continue
                if not os.path.exists(depth_path):
                    if accelerator.is_main_process:
                        print(f'[WARN] bokeh_failure depth not found: {depth_path}')
                    continue

                img = Image.open(all_in_focus_path).convert('RGB')
                img_np = np.array(img).astype(np.float32) / 255.0
                depth_np = load_depth_any(depth_path)
                if depth_np.ndim == 3:
                    depth_np = depth_np[..., 0]

                if args.assigned_size is not None:
                    img_np = center_crop_to_size(img_np, args.assigned_size, args.assigned_size)
                    depth_np = center_crop_to_size(depth_np, args.assigned_size, args.assigned_size)

                rel_path = sample_info['rel_path']
                sample_dir = os.path.join(args.output_root, rel_path)

                # If the sample has already been fully generated, skip it.
                if check_sample_complete(sample_dir, len(k_vals)):
                    skipped_count += 1
                    if accelerator.is_main_process and sample_idx % 50 == 0:
                        print(f'[SKIP] process {accelerator.process_index} skipping completed sample {sample_idx + 1}/{len(process_items)}: bokeh_failure/{rel_path}')
                    continue

                stack_dir = os.path.join(sample_dir, 'defocus_stack')
                ensure_dir(sample_dir)
                ensure_dir(stack_dir)

                source_rgb = all_in_focus_path
                source_depth_ref = depth_path
                sample_desc = f'bokeh_failure/{rel_path}'

                original_depth = sample_info.get('original_depth_path')
                if original_depth:
                    extra_meta['source_original_depth'] = original_depth
                similarity = sample_info.get('similarity_score')
                if similarity is not None:
                    try:
                        extra_meta['bokeh_failure_similarity'] = float(similarity)
                    except (TypeError, ValueError):
                        pass
            elif args.dataset == 'kitti_train':
                sequence_path, frame_id, camera_side, rgb_path, depth_index = item
                img = Image.open(rgb_path).convert('RGB')
                img_np = np.array(img).astype(np.float32) / 255.0
                if args.assigned_size is not None:
                    img_np = center_crop_to_size(img_np, args.assigned_size, args.assigned_size)

                try:
                    date_str, sequence_name = sequence_path.split('/')
                except ValueError:
                    date_str, sequence_name = sequence_path, 'unknown'
                frame_dirname = f'{frame_id:010d}'
                sample_dir = os.path.join(args.output_root, date_str, sequence_name, frame_dirname)

                if check_sample_complete(sample_dir, len(k_vals)):
                    skipped_count += 1
                    if accelerator.is_main_process and sample_idx % 50 == 0:
                        print(f'[SKIP] process {accelerator.process_index} skipping completed sample {sample_idx + 1}/{len(process_items)}: {sequence_path}#{frame_id:010d}')
                    continue

                stack_dir = os.path.join(sample_dir, 'defocus_stack')
                ensure_dir(stack_dir)

                depth_np = None
                source_depth_ref = None
                candidate_dirs: List[str] = []
                if kitti_train_depth_root:
                    candidate_dirs.append(os.path.join(kitti_train_depth_root, date_str, sequence_name, frame_dirname))
                candidate_dirs.append(sample_dir)

                for base_dir in candidate_dirs:
                    if not base_dir:
                        continue
                    depth_path = os.path.join(base_dir, 'depth.npy')
                    if os.path.exists(depth_path):
                        try:
                            depth_loaded = np.load(depth_path).astype(np.float32)
                            if depth_loaded.ndim == 3:
                                depth_loaded = depth_loaded[..., 0]
                            if args.assigned_size is not None:
                                depth_loaded = center_crop_to_size(depth_loaded, args.assigned_size, args.assigned_size)
                            depth_np = depth_loaded
                            source_depth_ref = depth_path
                            break
                        except Exception as exc:
                            if accelerator.is_main_process:
                                print(f'[WARN] failed to load depth.npy {depth_path}: {exc}')

                if depth_np is not None:
                    if not np.isfinite(depth_np).all():
                        finite_mask = np.isfinite(depth_np)
                        positive_mask = finite_mask & (depth_np > 0)
                        if np.any(positive_mask):
                            min_positive = float(np.min(depth_np[positive_mask]))
                            depth_np[~finite_mask] = 0
                            depth_np[depth_np <= 0] = min_positive if np.isfinite(min_positive) else 0.1
                        else:
                            depth_np = None
                            source_depth_ref = None
                    elif np.any(depth_np <= 0):
                        positive_mask = depth_np > 0
                        if np.any(positive_mask):
                            min_positive = float(np.min(depth_np[positive_mask]))
                            depth_np[~positive_mask] = min_positive

                if depth_np is None and accelerator.is_main_process and sample_idx % 50 == 0:
                    print(f'[WARN] no depth.npy found for KITTI training sample {sequence_path} frame {frame_id}')

                source_rgb = rgb_path
                sample_desc = f'{sequence_path}#{frame_id:010d}'
                source_mask_ref = None
            elif args.dataset == 'kitti_test':
                sequence_path, frame_id, camera_side, rgb_path, depth_index = item
                img = Image.open(rgb_path).convert('RGB')
                img_np = np.array(img).astype(np.float32) / 255.0
                if args.assigned_size is not None:
                    img_np = center_crop_to_size(img_np, args.assigned_size, args.assigned_size)

                try:
                    date_str, sequence_name = sequence_path.split('/')
                except ValueError:
                    date_str, sequence_name = sequence_path, 'unknown'
                sample_dir = os.path.join(args.output_root, date_str, sequence_name, f'{frame_id:010d}')

                # If the sample has already been fully generated, skip it.
                if check_sample_complete(sample_dir, len(k_vals)):
                    skipped_count += 1
                    if accelerator.is_main_process and sample_idx % 50 == 0:
                        print(f'[SKIP] process {accelerator.process_index} skipping completed sample {sample_idx + 1}/{len(process_items)}: {sequence_path}#{frame_id:010d}')
                    continue

                stack_dir = os.path.join(sample_dir, 'defocus_stack')
                ensure_dir(stack_dir)

                depth_np = None
                source_depth_ref = None
                if gt_depths_array is not None:
                    try:
                        depth_np = load_depth_from_npz_index(gt_depths_array, depth_index)
                        if depth_np.ndim == 3:
                            depth_np = depth_np[..., 0]
                        if args.assigned_size is not None:
                            depth_np = center_crop_to_size(depth_np, args.assigned_size, args.assigned_size)
                        depth_np = depth_np.astype(np.float32)
                        source_depth_ref = f'{args.kitti_gt_depths}[{depth_index}]'
                    except Exception as exc:
                        if accelerator.is_main_process:
                            print(f'[WARN] failed to read KITTI depth {sequence_path} frame {frame_id}: {exc}')

                source_rgb = rgb_path
                sample_desc = f'{sequence_path}#{frame_id:010d}'
                source_mask_ref = None
            else:
                if accelerator.is_main_process:
                    print(f'[WARN] unknown dataset: {args.dataset}, skip')
                continue

            stack_input_np = img_np
            if resize_enabled:
                stack_input_np = resize_preserve_aspect(img_np, resize_target, cv2.INTER_CUBIC)

            defocus_meta: Dict[str, float] = {}
            input_pil = Image.fromarray((stack_input_np * 255).astype(np.uint8))
            prompt_list = [build_prompt_from_template(prompt_template, k) for k in k_vals]

            generated_images = generate_i2i_bokeh_image(
                kontext_pipeline=kontext_pipeline,
                bokeh_adapter=bokeh_adapter,
                input_image=input_pil,
                prompt=prompt_list,
                k_values_batch=k_vals,
                K_max=args.K_max,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                device=accelerator.device,
                seed=args.seed + sample_idx * 1000,
                dtype=weight_dtype,
                vae_scale_factor=vae_scale_factor,
                assigned_size=args.assigned_size
            )

            if not isinstance(generated_images, (list, tuple)):
                generated_images = [generated_images]

            if len(generated_images) != len(k_vals):
                if accelerator.is_main_process:
                    print(f'[WARN] generated image count ({len(generated_images)}) does not match K count ({len(k_vals)}); truncating to match')

            for idx, K in enumerate(k_vals):
                if idx >= len(generated_images):
                    break
                generated_image = generated_images[idx]
                img_name = f'{idx}.png'
                generated_image.save(os.path.join(stack_dir, img_name))
                defocus_meta[str(idx)] = float(K)

            meta_path = os.path.join(stack_dir, 'stack_index.json')
            actual_size = [stack_input_np.shape[1], stack_input_np.shape[0]]
            source_size = [img_np.shape[1], img_np.shape[0]]
            meta_dict = {
                'ids': list(defocus_meta.keys()),
                'k_values': defocus_meta,
                'source_rgb': source_rgb,
                'source_depth': source_depth_ref,
                'size': actual_size,
                'source_size': source_size,
                'method': 'bokeh_diffusion_i2i',
                'prompt_template': prompt_template,
                'guidance_scale': args.guidance_scale,
                'num_inference_steps': args.num_inference_steps,
            }
            if source_mask_ref is not None:
                meta_dict['source_mask'] = source_mask_ref
            if source_intrinsics_ref is not None:
                meta_dict['source_intrinsics'] = source_intrinsics_ref
            for key, value in extra_meta.items():
                if value is not None:
                    meta_dict[key] = value

            with open(meta_path, 'w') as f:
                json.dump(meta_dict, f, indent=2)

            processed_count += 1
            if accelerator.is_main_process:
                print(f'[INFO] process {accelerator.process_index} finished sample {sample_idx + 1}/{len(process_items)}: {sample_desc}')

        except Exception as e:
            if accelerator.is_main_process:
                print(f'[ERROR] process {accelerator.process_index} failed to process sample: {e}')
            continue

    # Wait for every process (the balanced slicing avoids excessive waits).
    accelerator.wait_for_everyone()

    manifest_summary: Dict[str, Tuple[str, int]] = {}
    if accelerator.is_main_process:
        print('[INFO] generating JSONL manifest...')
        if args.dataset == 'nyuv2':
            manifest_summary = generate_nyuv2_manifests(args.output_root)
        else:
            manifest_summary = generate_generic_manifest(args.output_root, args.dataset)

    if accelerator.is_main_process:
        print('=' * 60)
        print(f'process {accelerator.process_index} stats:')
        print(f'  total samples: {len(process_items)}')
        print(f'  skipped (already done): {skipped_count}')
        print(f'  newly processed: {processed_count}')
        print(f'  failed/errored: {len(process_items) - skipped_count - processed_count}')
        print('=' * 60)
        if manifest_summary:
            print('manifest outputs:')
            for split_name, (path, count) in manifest_summary.items():
                print(f'  - {split_name}: {path} ({count} samples)')
        else:
            print(f'[WARN] no {args.dataset} manifest produced, check the output directory.')
        print('Done.')


if __name__ == '__main__':
    main()
    # Graceful shutdown (under Accelerate this tears down the process group and avoids resource-leak warnings).
    try:
        from accelerate import Accelerator
        Accelerator().end_training()
    except Exception:
        pass
