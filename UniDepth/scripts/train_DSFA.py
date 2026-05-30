"""Distributed training entrypoint for the UniDepthV2 DSFA variant.

The script wires ``UniDepthV2DSFA`` with the in-repo dataset stack, a cosine
LR/WD/beta scheduler trio, EMA, mixed-precision and DDP.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import uuid
from contextlib import nullcontext
from datetime import datetime as dt
from functools import partial
from pathlib import Path
from time import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.utils.data.distributed
import wandb
from wandb.integration.torch import wandb_torch
from torch import distributed as dist
from torch import optim
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

import unidepth.datasets as datasets
from unidepth.datasets import (
    ConcatDataset,
    DistributedSamplerNoDuplicate,
    collate_fn,
    get_weights,
)
from unidepth.models import UniDepthV2DSFA
from unidepth.ops.scheduler import CosineScheduler
from unidepth.utils.distributed import (
    barrier,
    create_local_process_group,
    is_main_process,
    local_broadcast_process_authkey,
    setup_multi_processes,
    sync_string_across_gpus,
    sync_tensor_across_gpus,
)
from unidepth.utils.ema_torch import (
    DummyExponentialMovingAverage,
    ExponentialMovingAverage,
)
from unidepth.utils.misc import calculate_mean_values, format_seconds
from unidepth.utils.validation import validate
from unidepth.utils.visualization import save_eval_visualization

EMA_INTERVAL = 10

# Map config "model.name" values to the concrete classes we expose.
MODEL_REGISTRY = {
    "UniDepthV2DSFA": UniDepthV2DSFA,
}


class BalancedGroupBatchSampler(torch.utils.data.Sampler[List[int]]):
    """Yield mini-batches that contain an equal number of samples from each group."""

    def __init__(
        self,
        groups: Sequence[Sequence[int]],
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        if not groups:
            raise ValueError("BalancedGroupBatchSampler requires at least one group.")
        self.groups: List[List[int]] = [list(group) for group in groups if len(group) > 0]
        if not self.groups:
            raise ValueError("BalancedGroupBatchSampler received empty groups.")
        self.num_groups = len(self.groups)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive for BalancedGroupBatchSampler.")
        if self.batch_size % self.num_groups != 0:
            raise ValueError(
                f"Batch size ({self.batch_size}) must be divisible by the number of groups "
                f"({self.num_groups}) to build balanced mini-batches."
            )
        self.samples_per_group = self.batch_size // self.num_groups
        self.shuffle = shuffle
        self._num_batches = self._compute_num_batches()

    def __iter__(self) -> Iterable[List[int]]:
        group_iters = [self._group_iterator(group) for group in self.groups]
        for _ in range(self._num_batches):
            batch: List[int] = []
            for group_iter in group_iters:
                for _ in range(self.samples_per_group):
                    batch.append(next(group_iter))
            if self.shuffle and len(batch) > 1:
                random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self._num_batches

    def _compute_num_batches(self) -> int:
        max_group = max(len(group) for group in self.groups)
        batches = math.ceil(max_group / max(1, self.samples_per_group))
        return max(1, batches)

    def _group_iterator(self, group: List[int]) -> Iterable[int]:
        while True:
            if self.shuffle:
                random.shuffle(group)
            for idx in group:
                yield idx


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEBUG_LOG_ENABLED = _env_flag("UNIDEPTH_DEBUG")


def _debug_print(*args, force: bool = False, **kwargs):
    if DEBUG_LOG_ENABLED or force:
        print(*args, **kwargs)


def _guard_wandb_none_grads():
    """Monkey-patch wandb hooks so gradients that are ``None`` are skipped safely."""

    def _install_guard(hook_cls):
        if hook_cls is None or getattr(hook_cls, "_supports_none_grad", False):
            return

        original_hook = getattr(hook_cls, "_hook_variable_gradient_stats", None)
        if original_hook is None:
            return

        def _safe_hook_variable_gradient_stats(self, var, name, log_track):
            if not isinstance(var, torch.Tensor):
                cls = type(var)
                raise TypeError(
                    f"Expected torch.Tensor, not {cls.__module__}.{cls.__name__}"
                )

            handle = self._hook_handles.get(name)
            if handle is not None and self._torch_hook_handle_is_valid(handle):
                raise ValueError(f'A hook has already been set under name "{name}"')

            def _callback(grad, local_log_track):
                if grad is None:
                    return
                if not wandb_torch.log_track_update(local_log_track):
                    return
                self.log_tensor_stats(grad.data, name)

            handle = var.register_hook(lambda grad: _callback(grad, log_track))
            self._hook_handles[name] = handle
            return handle

        hook_cls._hook_variable_gradient_stats = _safe_hook_variable_gradient_stats
        hook_cls._supports_none_grad = True

    _install_guard(getattr(wandb_torch, "GradientsHook", None))
    _install_guard(getattr(wandb_torch, "TorchHistory", None))


_guard_wandb_none_grads()


def aggregate_sync_losses(dict_: dict[str, torch.Tensor], device):
    keys = list(dict_.keys())
    values = torch.tensor(list(dict_.values()), device=device)
    keys = sync_string_across_gpus(keys, device)
    values = sync_tensor_across_gpus(values, dim=0).cpu().tolist()
    dict_ = calculate_mean_values(keys, values)
    return dict_


def _read_git_notes() -> str:
    """Best-effort wandb run notes derived from the current git checkout."""
    try:
        import git  # noqa: WPS433
    except Exception:
        return ""
    try:
        repo_folder = os.path.dirname(os.path.realpath(__file__))
        repo = git.Repo(repo_folder, search_parent_directories=True)
        current_head = repo.head if repo.head.is_detached else repo.active_branch
        return (
            f"MESSAGE: {current_head.commit.message} "
            f"HASH:{current_head.commit.hexsha} BRANCH:{current_head.name}"
        )
    except Exception as exc:
        print(f"problem reading git repo: {exc}")
        return ""


def main_worker(config: Dict[str, Any], args: argparse.Namespace):
    current_process = psutil.Process(os.getpid())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed = config["generic"]["seed"]

    # Resolve the checkpoint save directory.
    if args.save_dir is None:
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "exp",
            f"dsfa_{dt.now().strftime('%Y%m%d_%H%M%S')}",
        )
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir

    if not args.distributed:
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
    else:
        # Standard PyTorch DDP init: rely on env vars set by torchrun
        # (RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT).
        setup_multi_processes(config)

        args.rank = int(os.environ.get("RANK", 0))
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")

        create_local_process_group()
        local_broadcast_process_authkey()

        print(
            f"Start running DDP on: rank={args.rank}, local_rank={args.local_rank}, "
            f"world_size={args.world_size}"
        )

        # Per-GPU batch size when running distributed.
        config["training"]["batch_size"] = int(
            config["training"]["batch_size"] / args.world_size
        )
        dist.barrier()

    # Validation visualization directory (rank-aware creation).
    vis_dir = Path(save_dir) / "validation_visualizations"
    if args.rank == 0:
        vis_dir.mkdir(parents=True, exist_ok=True)
    if args.distributed:
        barrier()

    # Per-rank seed offset so workers do not draw the same samples.
    seed = seed + args.rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    batch_size = config["training"]["batch_size"]
    if is_main_process():
        print("Config: ", args.config_file)
        print(
            f"Torch version:{torch.__version__}, cuda:{torch.version.cuda}, "
            f"cudnn:{torch.backends.cudnn.version()}, threads:{torch.get_num_threads()}"
        )
        print("BatchSize per GPU: ", batch_size)
        print(
            f"Divided into {config['training']['nsteps_accumulation_gradient']} accumulation step"
        )

    # ----------------------------- Model ----------------------------------
    model_name = config["model"]["name"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name '{model_name}'. Supported: {sorted(MODEL_REGISTRY)}"
        )
    model = MODEL_REGISTRY[model_name](config).to(device)
    model.eval()
    print(f"MODEL: {model.__class__.__name__} at {model.device}")
    torch.cuda.empty_cache()
    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # NB: DDP with find_unused_parameters=True is incompatible with gradient
        # checkpointing; CheckpointFunction marks downstream params as unused, then
        # marks them ready again inside its own forward, which triggers an assert
        # in the DDP Reducer.
        ddp_find_unused = config["training"].get("find_unused_parameters", False)
        if ddp_find_unused and is_main_process():
            print("DDP: enabling find_unused_parameters=True")
        model = DistributedDataParallel(
            model,
            find_unused_parameters=ddp_find_unused,
            device_ids=[device],
            output_device=device,
        )

    # --------------------- Load pretrained weights ------------------------
    # The pretrained payload must be loaded *before* building the optimizer:
    # ``get_params`` relies on ``_new_param_names`` / ``_pretrained_param_names``
    # which ``load_pretrained`` populates.
    training_cfg = config.get("training", {})
    enable_mse_loss = bool(training_cfg.get("enable_mse_loss", False))
    mse_loss_weight = float(training_cfg.get("mse_loss_weight", 1.0))
    resume_checkpoint = getattr(args, "resume_checkpoint", None) or training_cfg.get(
        "resume_checkpoint"
    )
    base_pretrained_path = training_cfg.get("pretrained")

    ddp_model = model.module if args.distributed else model
    step = 0
    is_resume_checkpoint = False
    pretrained_payload: Optional[dict[str, Any]] = None

    if resume_checkpoint:
        if not os.path.isfile(resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")

        # Load base pretrained metadata first so the new/pretrained parameter split
        # is preserved and differential LR groups can be restored.
        if base_pretrained_path and os.path.isfile(base_pretrained_path):
            if is_main_process():
                print(f"\n>>> Loading pretrained metadata: {base_pretrained_path}")
            try:
                ddp_model.load_pretrained(base_pretrained_path)
                if is_main_process():
                    print(">>> Pretrained metadata loaded\n")
            except Exception as exc:
                if is_main_process():
                    print(f">>> WARNING: failed to load pretrained metadata ({exc})")

        if is_main_process():
            print(f"\n>>> Loading training checkpoint: {resume_checkpoint}")
        pretrained_payload = torch.load(
            resume_checkpoint, map_location="cpu", weights_only=False
        )

        if {"model", "optimizer", "step"}.issubset(pretrained_payload.keys()):
            is_resume_checkpoint = True
            if is_main_process():
                print(">>> Detected full training checkpoint, restoring state...")
            ddp_model.load_state_dict(pretrained_payload["model"])
            step = pretrained_payload["step"]
            if is_main_process():
                print(f">>> Resuming from step {step}\n")
        else:
            if is_main_process():
                print(">>> No full training state found, loading weights only (strict=False)")
            ddp_model.load_pretrained(resume_checkpoint)
    elif base_pretrained_path:
        if not os.path.isfile(base_pretrained_path):
            raise FileNotFoundError(f"Pretrained weights not found: {base_pretrained_path}")
        if is_main_process():
            print(f"\n>>> Loading pretrained model: {base_pretrained_path}")
        ddp_model.load_pretrained(base_pretrained_path)
        if is_main_process():
            print(">>> Pretrained weights loaded\n")

    # ---------------------------- Optimizer -------------------------------
    f16 = config["training"].get("f16", False)
    clipping = config["training"].get("clipping", None)

    params = ddp_model.get_params(config)
    optimizer = optim.AdamW(params)

    scaler = torch.amp.GradScaler("cuda", enabled=f16)

    ema_class = (
        ExponentialMovingAverage
        if config["training"]["ema"]
        else DummyExponentialMovingAverage
    )
    ema_handle = ema_class(
        model.parameters(),
        1 - (1 - 0.9995) * EMA_INTERVAL,
        update_after_step=75000 // EMA_INTERVAL,
        tau=20000 // EMA_INTERVAL,
    )
    setattr(ema_handle, "num_updates", step // EMA_INTERVAL)

    # When resuming, also restore optimizer / scaler / EMA state.
    if is_resume_checkpoint:
        try:
            optimizer.load_state_dict(pretrained_payload["optimizer"])
            if is_main_process():
                print(">>> Optimizer state restored")
        except Exception as exc:
            if is_main_process():
                print(f">>> WARNING: failed to restore optimizer state: {exc}")

        try:
            scaler.load_state_dict(pretrained_payload["scaler"])
            if is_main_process():
                print(">>> Scaler state restored")
        except Exception as exc:
            if is_main_process():
                print(f">>> WARNING: failed to restore scaler state: {exc}")

        if pretrained_payload and pretrained_payload.get("ema") is not None:
            try:
                ema_handle.load_state_dict(pretrained_payload["ema"])
                if is_main_process():
                    print(">>> EMA state restored")
            except Exception as exc:
                if is_main_process():
                    print(f">>> WARNING: failed to restore EMA state: {exc}")

    # ----------------------------- Generic --------------------------------
    resize_method = config["data"].get("resize_method", "hard")
    crop = config["data"].get("crop", "garg")
    augmentations_db = config["data"].get("augmentations", {})
    image_shape = config["data"]["image_shape"]
    nsteps_accumulation_gradient = config["training"]["nsteps_accumulation_gradient"]
    batch_size = config["training"]["batch_size"] // config["data"]["num_copies"]

    is_shell = int(os.environ.get("SHELL_JOB", 0))
    run_id = sync_string_across_gpus(
        [f"{dt.now().strftime('%d-%h_%H-%M')}-{uuid.uuid4()}"], device
    )[0]

    # wandb is always initialised unless WANDB_MODE=disabled is set in the env.
    if is_main_process():
        notes = _read_git_notes()

        # Restore the global batch size in the config we hand to wandb so the
        # logged value matches the user-facing configuration.
        if args.distributed:
            config["training"]["batch_size"] = (
                config["training"]["batch_size"]
                * args.world_size
                * config["data"]["num_copies"]
            )
        wandb.init(
            project="UniDepth",
            name=run_id,
            config=config,
            tags=None,
            notes=notes,
            dir=os.environ.get("WANDB_HOME", os.environ.get("TMPDIR", "/tmp")),
        )
        wandb.watch(model)

    # ----------------------------- Dataset --------------------------------
    train_datasets, val_datasets, dims = {}, {}, 0
    if is_main_process():
        print("Loading training datasets...")
    for dataset in config["data"]["train_datasets"]:
        assert hasattr(datasets, dataset), f"{dataset} not a custom dataset"
        train_dataset: datasets.BaseDataset = getattr(datasets, dataset)

        # Allow per-dataset manifest overrides keyed by the lower-case dataset name.
        dataset_manifest_key = f"{dataset.lower()}_manifest_path"
        dataset_manifest_list_key = f"{dataset.lower()}_manifest_paths"
        manifest_path = config["data"].get(dataset_manifest_key, None)
        manifest_paths = config["data"].get(dataset_manifest_list_key, None)

        dataset_kwargs = {
            "image_shape": image_shape,
            "split_file": train_dataset.train_split,
            "test_mode": False,
            "crop": crop,
            "augmentations_db": augmentations_db,
            "normalize": config["data"].get("normalization", "imagenet"),
            "resize_method": resize_method,
            "mini": 1.0,
            "num_copies": config["data"]["num_copies"],
            "num_frames": 1,
            "fps_range": [1, 30],
            "data_root": config["data"]["data_root"],
        }

        defocus_indices = config["data"].get("defocus_stack_indices")
        if defocus_indices is not None:
            dataset_kwargs["defocus_stack_indices"] = defocus_indices

        if dataset.upper() == "NYUV2DEPTH":
            dataset_kwargs.setdefault("manifest_depth_mode", "auto")

        if manifest_paths:
            dataset_kwargs["manifest_paths"] = manifest_paths
            if is_main_process():
                print(f"  {dataset} using manifests:")
                for path in manifest_paths:
                    print(f"    - {path}")
        elif manifest_path:
            dataset_kwargs["manifest_path"] = manifest_path
            if is_main_process():
                print(f"  {dataset} using manifest: {manifest_path}")

        if dataset.upper() == "HYPERSIM":
            dataset_kwargs["manifest_split"] = None
            camera_meta_path = config["data"].get("hypersim_camera_metadata_path")
            if camera_meta_path:
                dataset_kwargs["camera_metadata_path"] = camera_meta_path

        train_datasets[dataset] = train_dataset(**dataset_kwargs)

        # Report dataset size when the underlying memory-mapped buffers are
        # available, otherwise just print the sample count.
        if hasattr(train_datasets[dataset], "dataset") and hasattr(
            train_datasets[dataset].dataset, "_addr"
        ):
            dim = (
                train_datasets[dataset].dataset._addr.numel() * 8
                + train_datasets[dataset].dataset._lst.numel()
            ) / (2**20)
            if hasattr(train_datasets[dataset], "sequences"):
                dim += (
                    train_datasets[dataset].sequences._addr.numel() * 8
                    + train_datasets[dataset].sequences._lst.numel()
                ) / (2**20)
            if is_main_process():
                print(f"{dataset}: {dim:.1f}MB")
            dims += dim
        else:
            if is_main_process():
                print(f"{dataset}: {len(train_datasets[dataset])} samples")

    if is_main_process():
        print(f"All training datasets loaded, with total size: {dims:.1f}MB")

    barrier()

    assert batch_size % nsteps_accumulation_gradient == 0
    batch_chunk = batch_size // nsteps_accumulation_gradient

    train_dataset = ConcatDataset(
        [t for t in train_datasets.values()],
        shape_constraints=config["data"]["augmentations"]["shape_constraints"],
    )

    manifest_groups_global: List[List[int]] = []
    covered_indices: set[int] = set()
    offset = 0
    for dataset_obj in train_datasets.values():
        group_indices = getattr(dataset_obj, "manifest_group_indices", None)
        if group_indices and len(group_indices) > 1:
            for group in group_indices:
                global_group = [offset + idx for idx in group]
                manifest_groups_global.append(global_group)
                covered_indices.update(global_group)
        offset += len(dataset_obj)

    balanced_sampler: Optional[BalancedGroupBatchSampler] = None
    if manifest_groups_global and len(manifest_groups_global) > 1:
        total_indices = len(covered_indices)
        if total_indices == len(train_dataset):
            if batch_size % len(manifest_groups_global) != 0:
                raise ValueError(
                    f"Training batch size ({batch_size}) must be divisible by the number "
                    f"of HyperSim manifests ({len(manifest_groups_global)}) to enable 1:1 sampling."
                )
            balanced_sampler = BalancedGroupBatchSampler(
                manifest_groups_global, batch_size
            )
            if is_main_process():
                group_sizes = ", ".join(
                    str(len(group)) for group in manifest_groups_global
                )
                print(
                    f"Enabled balanced HyperSim manifest sampling "
                    f"({len(manifest_groups_global)} groups) with group sizes: {group_sizes}"
                )
        else:
            if is_main_process():
                print(
                    "Skipping balanced HyperSim sampling because manifest groups do not "
                    "cover the complete dataset."
                )

    # Validation datasets (optional).
    if config["data"]["val_datasets"]:
        if is_main_process():
            print("Loading validation datasets...")
        for dataset in config["data"]["val_datasets"]:
            val_dataset: datasets.BaseDataset = getattr(datasets, dataset)

            dataset_val_manifest_key = f"{dataset.lower()}_val_manifest_path"
            dataset_manifest_key = f"{dataset.lower()}_manifest_path"
            manifest_path = config["data"].get(
                dataset_val_manifest_key, None
            ) or config["data"].get(dataset_manifest_key, None)

            dataset_kwargs = {
                "image_shape": image_shape,
                "split_file": val_dataset.test_split,
                "test_mode": True,
                "crop": crop,
                "augmentations_db": augmentations_db,
                "normalize": config["data"].get("normalization", "imagenet"),
                "resize_method": resize_method,
                "num_frames": 1,
                "mini": 1.0,
                "data_root": config["data"]["data_root"],
            }

            defocus_indices = config["data"].get("defocus_stack_indices")
            if defocus_indices is not None:
                dataset_kwargs["defocus_stack_indices"] = defocus_indices

            if dataset.upper() == "NYUV2DEPTH":
                dataset_kwargs.setdefault("manifest_depth_mode", "auto")

            if manifest_path:
                dataset_kwargs["manifest_path"] = manifest_path
                if is_main_process():
                    print(f"  {dataset} validation using manifest: {manifest_path}")

            if dataset.upper() == "HYPERSIM":
                dataset_kwargs["manifest_split"] = config["data"].get(
                    "hypersim_val_manifest_split", "test"
                )
                camera_meta_path = config["data"].get("hypersim_camera_metadata_path")
                if camera_meta_path:
                    dataset_kwargs["camera_metadata_path"] = camera_meta_path

            val_datasets[dataset] = val_dataset(**dataset_kwargs)
    else:
        if is_main_process():
            print("No validation datasets specified, skipping validation.")

    # Distributed samplers pinned to rank.
    if balanced_sampler is not None:
        train_sampler = balanced_sampler
        if args.distributed and val_datasets:
            valid_samplers = {
                k: DistributedSamplerNoDuplicate(
                    v,
                    num_replicas=args.world_size,
                    rank=args.rank,
                    shuffle=False,
                    drop_last=False,
                )
                for k, v in val_datasets.items()
            }
        else:
            valid_samplers = (
                {k: SequentialSampler(v) for k, v in val_datasets.items()}
                if val_datasets
                else {}
            )
    else:
        if args.distributed:
            weights, num_samples = get_weights(
                train_datasets, config["data"]["sampling"]
            )
            train_sampler = torch.utils.data.WeightedRandomSampler(
                weights, num_samples, replacement=True
            )
            if val_datasets:
                valid_samplers = {
                    k: DistributedSamplerNoDuplicate(
                        v,
                        num_replicas=args.world_size,
                        rank=args.rank,
                        shuffle=False,
                        drop_last=False,
                    )
                    for k, v in val_datasets.items()
                }
            else:
                valid_samplers = {}
        else:
            train_sampler = RandomSampler(train_dataset)
            valid_samplers = (
                {k: SequentialSampler(v) for k, v in val_datasets.items()}
                if val_datasets
                else {}
            )

        train_sampler = torch.utils.data.BatchSampler(
            train_sampler, batch_size=batch_size, drop_last=True
        )

    val_batch_size = 1
    num_workers = 0
    train_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=partial(collate_fn, is_batched=True),
        persistent_workers=num_workers > 0,
    )
    val_loaders = {
        name_dataset: DataLoader(
            dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=valid_samplers[name_dataset],
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, is_batched=False),
        )
        for name_dataset, dataset in val_datasets.items()
    }

    # ----------------------------- Schedulers ------------------------------
    scheduler_wd = CosineScheduler(
        optimizer,
        key="weight_decay",
        init_value=config["training"]["wd"],
        base_value=config["training"]["wd"],
        final_value=config["training"]["wd_final"],
        warmup_iters=0,
        total_iters=config["training"]["n_iters"],
        step_init=step - 1,
    )
    scheduler_lr = CosineScheduler(
        optimizer,
        key="lr",
        init_value=config["training"]["lr"] * config["training"].get("lr_warmup", 1.0),
        final_value=config["training"]["lr_final"],
        warmup_iters=config["training"]["warmup_iters"],
        total_iters=config["training"]["n_iters"],
        step_init=step - 1,
    )
    scheduler_betas = CosineScheduler(
        optimizer,
        key="betas",
        init_value=0.95 if config["training"].get("cycle_betas", True) else 0.9,
        base_value=0.85 if config["training"].get("cycle_betas", True) else 0.9,
        final_value=0.95 if config["training"].get("cycle_betas", True) else 0.9,
        warmup_iters=config["training"]["warmup_iters"],
        total_iters=config["training"]["n_iters"],
        step_init=step - 1,
    )

    # bf16 has some resolution issues for SILog at small depths, so we default
    # to fp16. The mem-efficient SDPA path is disabled because the supposed
    # memory win disappears under fp16 + checkpointing.
    dtype = torch.float16
    context = torch.autocast(device_type="cuda", dtype=dtype, enabled=f16)
    optimizer.zero_grad(set_to_none=True)

    # ----------------------------- Training -------------------------------
    # Note: if any of the encoder layers are frozen, gradient checkpointing on
    # the next layer must still pass requires_grad=True on the input or DDP
    # will mis-mark parameters as unused.
    ddp_model.train()
    start = time()
    n_steps = config["training"]["n_iters"]
    init_steps = int(step)
    track_pbar = is_shell

    if is_main_process():
        print("Is a shell job?", is_shell)
        print("Use dtype:", dtype if f16 else torch.float32)
        print(
            f'Train for {config["training"]["n_iters"]} steps, validate every '
            f'{config["training"]["validation_interval"]} steps'
        )
        print(f"START with {num_workers} workers")
        if track_pbar:
            pbar = tqdm(total=n_steps - init_steps)

    track_losses: Dict[str, torch.Tensor] = {}
    system_memory = dict(psutil.virtual_memory()._asdict())["available"] / 2**30
    cpid_memory = current_process.memory_info()[0] / 2.0**30
    gpu_mem = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 2**30

    while True:
        for j, batches in enumerate(train_loader):

            system_memory = (
                0.99 * system_memory
                + 0.01 * dict(psutil.virtual_memory()._asdict())["available"] / 2**30
            )
            cpid_memory = (
                0.99 * cpid_memory + 0.01 * current_process.memory_info()[0] / 2.0**30
            )
            gpu_mem = (
                0.99 * gpu_mem
                + 0.01
                * (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0])
                / 2**30
            )
            if j % 1000 == 0 and is_main_process():
                _debug_print(f"System information at step {j}")
                _debug_print(f"System-wide RAM available: {system_memory:.2f}GB")
                _debug_print(f"CPU utilization: {psutil.cpu_percent(interval=None)}%")
                _debug_print(f"GPU memory utilized: {gpu_mem:.2f}GB")

            batches["data"] = {
                k: v.to(model.device, non_blocking=True)
                for k, v in batches["data"].items()
            }

            accumulated_losses_dict: Dict[str, List[torch.Tensor]] = {}
            skip_current_step = False

            for idx in range(nsteps_accumulation_gradient):
                batch: Dict[str, Any] = {}
                batch_slice = slice(idx * batch_chunk, (idx + 1) * batch_chunk)
                batch["data"] = {k: v[batch_slice] for k, v in batches["data"].items()}
                batch["img_metas"] = batches["img_metas"][batch_slice]

                # Drop the temporal axis from the dataloader (always 1 here).
                batch["data"] = {
                    k: torch.flatten(v, 0, 1) for k, v in batch["data"].items()
                }
                batch["img_metas"] = [
                    {k: v for k, v in meta.items() if isinstance(v, list)}
                    for meta in batch["img_metas"]
                ]

                # Sanitise inputs: clean non-finite pixels and recompute masks.
                data_dict = batch["data"]
                for key in ["image", "defocus_stack"]:
                    if key in data_dict:
                        tensor = data_dict[key].float()
                        if not torch.all(torch.isfinite(tensor)):
                            if is_main_process() and idx == 0:
                                nan_count = (~torch.isfinite(tensor)).sum().item()
                                print(
                                    f"Step {step}: detected {nan_count} non-finite values "
                                    f"in {key}, cleaned"
                                )
                            tensor = torch.nan_to_num(
                                tensor, nan=0.0, posinf=1.0, neginf=0.0
                            )
                            tensor = torch.clamp(
                                tensor,
                                0.0,
                                255.0 if key in ["image", "defocus_stack"] else 1e6,
                            )
                            data_dict[key] = tensor.to(data_dict[key].dtype)

                if "depth" in data_dict:
                    depth_tensor = data_dict["depth"].float()
                    finite_mask = torch.isfinite(depth_tensor)
                    positive_mask = depth_tensor > 1e-6
                    valid_depth_mask = finite_mask & positive_mask

                    if not torch.all(valid_depth_mask):
                        depth_tensor = torch.where(
                            valid_depth_mask,
                            depth_tensor,
                            torch.zeros_like(depth_tensor),
                        )
                        data_dict["depth"] = depth_tensor

                    if "depth_mask" in data_dict:
                        data_dict["depth_mask"] = (
                            data_dict["depth_mask"].bool() & valid_depth_mask
                        )
                    else:
                        data_dict["depth_mask"] = valid_depth_mask

                    if "validity_mask" in data_dict:
                        data_dict["validity_mask"] = (
                            data_dict["validity_mask"].bool() & valid_depth_mask
                        )
                    else:
                        data_dict["validity_mask"] = valid_depth_mask

                with (
                    model.no_sync()
                    if idx < nsteps_accumulation_gradient - 1
                    else nullcontext()
                ):
                    with context:
                        preds, losses = model(batch["data"], batch["img_metas"])

                    # Optional auxiliary MSE term on metric depth.
                    if enable_mse_loss and isinstance(preds, dict):
                        depth_pred = preds.get("depth")
                        depth_gt = batch["data"].get("depth")
                        if depth_pred is not None and depth_gt is not None:
                            depth_mask = batch["data"].get("depth_mask")
                            valid_mask = (
                                depth_mask.bool()
                                if depth_mask is not None
                                else torch.ones_like(depth_gt, dtype=torch.bool)
                            )
                            valid_mask = valid_mask & torch.isfinite(depth_gt)

                            # Always insert the entry so the per-chunk loss dicts
                            # remain structurally identical during accumulation.
                            if valid_mask.any():
                                diff = depth_pred - depth_gt
                                mse_loss = (diff.pow(2)[valid_mask]).mean()
                                losses["opt"]["metric_depth_mse"] = (
                                    mse_loss_weight * mse_loss
                                )
                            else:
                                losses["opt"]["metric_depth_mse"] = torch.tensor(
                                    0.0,
                                    device=depth_pred.device,
                                    dtype=depth_pred.dtype,
                                )
                        else:
                            losses["opt"]["metric_depth_mse"] = torch.tensor(
                                0.0, device=model.device, dtype=torch.float32
                            )

                    # If SILog is exactly zero on this mini-batch we drop it
                    # from the optimisation set to avoid contaminating the grad.
                    silog_is_zero = False
                    silog_key_in_batch = None
                    for k, v in losses["opt"].items():
                        if "silog" in k.lower():
                            silog_key_in_batch = k
                            if v.abs() < 1e-7:
                                silog_is_zero = True
                            break

                    if silog_is_zero and silog_key_in_batch is not None:
                        silog_value = losses["opt"].pop(silog_key_in_batch)
                        if is_main_process() and idx == 0:
                            print(
                                f"Step {step}: SILog=0.00000 (batch_chunk={idx}), "
                                f"skipping SILog loss; other losses untouched"
                            )

                    # Capture per-chunk loss values for logging.
                    losses_dict = {
                        k: v.detach()
                        for loss in losses.values()
                        for k, v in loss.items()
                    }
                    if silog_is_zero and silog_key_in_batch is not None:
                        losses_dict[silog_key_in_batch] = silog_value.detach()

                    for k, v in losses_dict.items():
                        accumulated_losses_dict.setdefault(k, []).append(v)

                    # Combined optimisation loss (after the SILog pruning).
                    if len(losses["opt"]) > 0:
                        loss = sum(losses["opt"].values()) / nsteps_accumulation_gradient

                        if not torch.isfinite(loss):
                            if is_main_process():
                                print(
                                    f"Step {step}, batch_chunk {idx}: non-finite loss="
                                    f"{loss.item():.6f}, skipping this chunk's backward"
                                )
                                print(
                                    f"   Loss composition: "
                                    f"{[(k, v.item()) for k, v in losses['opt'].items()]}"
                                )
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            skip_current_step = True
                        else:
                            try:
                                scaler.scale(loss).backward()
                            except RuntimeError as exc:
                                if "CUDA" in str(exc) or "CUBLAS" in str(exc):
                                    if is_main_process():
                                        print(
                                            f"Step {step}, batch_chunk {idx}: "
                                            f"CUDA error: {exc}"
                                        )
                                        print(
                                            "   Clearing CUDA cache and skipping this step..."
                                        )
                                    torch.cuda.empty_cache()
                                    if torch.cuda.is_available():
                                        torch.cuda.synchronize()
                                    optimizer.zero_grad(set_to_none=True)
                                    skip_current_step = True
                                else:
                                    raise
                    else:
                        if is_main_process() and idx == 0:
                            print(
                                f"Step {step}: all losses were dropped, skipping backward"
                            )
                        skip_current_step = True

            # Optimizer step (skipped if we marked the current iteration as bad).
            if not skip_current_step:
                if clipping is not None:
                    scaler.unscale_(optimizer)
                    has_inf_grad = False
                    for param in model.parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            has_inf_grad = True
                            break

                    if has_inf_grad:
                        if is_main_process():
                            print(
                                f"Step {step}: non-finite grads detected, skipping update"
                            )
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    scaler.step(optimizer)
                    scaler.update()

                scheduler_wd.step()
                scheduler_lr.step()
                scheduler_betas.step()
                optimizer.zero_grad(set_to_none=True)
                if step % EMA_INTERVAL == 0:
                    ema_handle.update()
            else:
                if is_main_process():
                    print(f"Step {step}: skipping parameter update")
                optimizer.zero_grad(set_to_none=True)
                # Keep schedulers in sync with the wall-clock step counter.
                scheduler_wd.step()
                scheduler_lr.step()
                scheduler_betas.step()

            # Aggregate per-chunk losses and update the rolling EMA.
            avg_losses_dict: Dict[str, torch.Tensor] = {}
            for k, v in accumulated_losses_dict.items():
                if len(v) > 0:
                    avg_losses_dict[k] = torch.stack(v).mean()
                else:
                    avg_losses_dict[k] = torch.tensor(0.0, device=device)

            track_losses.update(
                {
                    k: 0.99 * track_losses.get(k, v)
                    + 0.01 * torch.nan_to_num(v, nan=1e5, posinf=1e5, neginf=1e5)
                    for k, v in avg_losses_dict.items()
                }
            )

            if is_main_process() and track_pbar:
                pbar.update(1)

            step += 1

            # Logging
            track_losses = aggregate_sync_losses(track_losses, device=model.device)
            if is_main_process():
                try:
                    wandb.log(
                        {
                            **{f"Train/{k}": v for k, v in track_losses.items()},
                            **{f"Train/lr": scheduler_lr.get()[-1]},
                            **{f"Train/wd": scheduler_wd.get()[-2]},
                            **{f"Train/scale_f16": scaler.get_scale()},
                        },
                        step=step,
                    )
                except Exception as exc:
                    print("Not logging loss because of:", exc)

                log_loss_dict = {f"Train/{k}": v for k, v in track_losses.items()}
                elapsed = int(time() - start)
                eta = int(elapsed * (n_steps - step) / max(1, step - init_steps))
                print(
                    f"Loss at {step}/{n_steps} "
                    f"[{format_seconds(elapsed)}<{format_seconds(eta)}]:"
                )
                print(", ".join([f"{k}: {v:.5f}" for k, v in log_loss_dict.items()]))

            # Checkpointing
            is_save_step = step % args.save_interval == 0 and step > 0
            is_last_step = step >= config["training"]["n_iters"]
            if (is_save_step or is_last_step) and is_main_process():
                checkpoint_path = os.path.join(
                    args.save_dir, f"checkpoint_step_{step}.pth"
                )
                print(f"\n>>> Saving checkpoint to: {checkpoint_path}")

                checkpoint = {
                    "step": step,
                    "model": ddp_model.state_dict()
                    if args.distributed
                    else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "ema": ema_handle.state_dict()
                    if hasattr(ema_handle, "state_dict")
                    else None,
                    "config": config,
                    "scheduler_lr": scheduler_lr.get(),
                    "scheduler_wd": scheduler_wd.get(),
                    "scheduler_betas": scheduler_betas.get(),
                }
                torch.save(checkpoint, checkpoint_path)

                latest_path = os.path.join(args.save_dir, "latest.pth")
                torch.save(checkpoint, latest_path)
                print(f">>> Checkpoint saved (step {step})\n")

            if is_save_step or is_last_step:
                barrier()

            # Validation (only when val loaders are configured).
            is_last_step = step >= config["training"]["n_iters"]
            is_validation = step % config["training"]["validation_interval"] == 0
            if (is_last_step or is_validation) and val_loaders:
                torch.cuda.empty_cache()
                barrier()
                if is_main_process():
                    print(f"Validation at {step}th step...")

                def validation_visualization_callback(
                    batch_inputs, preds, sample_index, batch_index, **kwargs
                ):
                    """Save an RGB / GT / Pred triplet for a single sample per rank."""
                    try:
                        rgb_tensor = batch_inputs["image"][sample_index].detach().cpu()
                        if rgb_tensor.dtype == torch.uint8:
                            rgb_np = rgb_tensor.float() / 255.0
                        else:
                            # Assume ImageNet-normalised float input.
                            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                            rgb_np = rgb_tensor * std + mean

                        if rgb_np.dim() == 3:
                            rgb_np = rgb_np.permute(1, 2, 0).numpy()
                        else:
                            rgb_np = rgb_np.numpy()
                        rgb_np = np.clip(rgb_np, 0.0, 1.0)

                        gt_depth = batch_inputs.get("depth")
                        if gt_depth is not None:
                            gt_np = (
                                gt_depth[sample_index].detach().cpu().squeeze().numpy()
                            )
                        else:
                            gt_np = None

                        pred_depth = preds.get("depth")
                        if pred_depth is not None:
                            pred_np = (
                                pred_depth[sample_index]
                                .detach()
                                .cpu()
                                .squeeze()
                                .numpy()
                            )
                        else:
                            return

                        depth_mask = batch_inputs.get("depth_mask")
                        if depth_mask is not None:
                            mask_np = (
                                depth_mask[sample_index]
                                .detach()
                                .cpu()
                                .squeeze()
                                .numpy()
                            )
                        else:
                            mask_np = None

                        output_filename = (
                            f"step{step:06d}_rank{args.rank}_batch{batch_index:03d}.png"
                        )
                        output_path = vis_dir / output_filename

                        save_eval_visualization(
                            rgb=rgb_np,
                            pred_depth=pred_np,
                            gt_depth=gt_np,
                            mask=mask_np,
                            out_path=output_path,
                            align=True,
                            cmap="magma_r",
                        )

                        if args.rank == 0 and batch_index == 0:
                            print(
                                f"  [Rank {args.rank}] "
                                f"Saved validation visualization: {output_path}"
                            )

                    except Exception as exc:
                        print(f"  [Rank {args.rank}] Visualization callback failed: {exc}")

                ddp_model.eval()
                start_validation = time()
                with torch.no_grad(), ema_handle.average_parameters():
                    validate(
                        model,
                        test_loaders=val_loaders,
                        step=step,
                        context=context,
                        visualize_fn=validation_visualization_callback,
                        max_visualizations=1,
                    )

                if is_main_process():
                    print(f"Elapsed: {format_seconds(int(time() - start_validation))}")
                ddp_model.train()
                torch.cuda.empty_cache()

            if step >= config["training"]["n_iters"]:
                if is_main_process() and track_pbar:
                    pbar.close()
                wandb.finish(0)
                dist.destroy_process_group()
                return 0


if __name__ == "__main__":
    # Avoid Triton cache clashes on multi-node Slurm jobs.
    if "SLURM_PROCID" in os.environ:
        os.environ["TRITON_CACHE_DIR"] = "/tmp"

    parser = argparse.ArgumentParser(
        description="UniDepthV2 DSFA training", conflict_handler="resolve"
    )
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--master-port", type=str)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument(
        "--fusion-layers",
        type=str,
        help="Comma separated encoder stage indices for DSFA fusion",
    )
    parser.add_argument(
        "--use-cls-token",
        action="store_true",
        help="Enable CLS token prefix conditioning for stack fusion",
    )
    parser.add_argument(
        "--hypersim-manifest",
        type=str,
        help="HyperSim manifest JSONL path (overrides config and env)",
    )
    parser.add_argument(
        "--hypersim-manifests",
        type=str,
        help=(
            "Comma separated HyperSim manifest JSONL paths for balanced sampling "
            "(overrides config and env)"
        ),
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5000,
        help="Save checkpoint every N steps (default: 5000)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (default: exp/dsfa_TIMESTAMP)",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to a training checkpoint for resuming (overrides config)",
    )
    parser.add_argument(
        "--enable-mse-loss",
        action="store_true",
        help="Enable auxiliary metric-depth MSE loss term",
    )

    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)

    training_cfg = config.setdefault("training", {})
    training_cfg.setdefault("enable_mse_loss", False)
    training_cfg.setdefault("mse_loss_weight", 1.0)
    if args.enable_mse_loss:
        training_cfg["enable_mse_loss"] = True
    args.enable_mse_loss = training_cfg["enable_mse_loss"]
    resume_ckpt = (
        args.resume_checkpoint
        or os.environ.get("UNIDEPTH_RESUME_CKPT")
        or training_cfg.get("resume_checkpoint")
    )
    if resume_ckpt:
        resume_ckpt = os.path.expanduser(resume_ckpt)
        training_cfg["resume_checkpoint"] = resume_ckpt
        print(f"Resuming from checkpoint: {resume_ckpt}")
        if args.save_dir is None:
            args.save_dir = os.path.dirname(resume_ckpt)
        args.resume_checkpoint = resume_ckpt

    # Resolve dataset manifest paths.
    # Priority: CLI > environment variable > config file value.
    data_cfg = config.setdefault("data", {})

    hypersim_manifest_paths: Optional[List[str]] = None
    if args.hypersim_manifests:
        hypersim_manifest_paths = [
            os.path.expanduser(path.strip())
            for path in args.hypersim_manifests.split(",")
            if path.strip()
        ]
    elif os.environ.get("HYPERSIM_MANIFEST_PATHS"):
        hypersim_manifest_paths = [
            os.path.expanduser(path.strip())
            for path in os.environ["HYPERSIM_MANIFEST_PATHS"].split(os.pathsep)
            if path.strip()
        ]
    elif isinstance(data_cfg.get("hypersim_manifest_paths"), (list, tuple)):
        hypersim_manifest_paths = [
            os.path.expanduser(os.fspath(path))
            for path in data_cfg["hypersim_manifest_paths"]
        ]

    if hypersim_manifest_paths:
        data_cfg["hypersim_manifest_paths"] = hypersim_manifest_paths
        data_cfg["hypersim_manifest_path"] = hypersim_manifest_paths[0]
        print("Using HyperSim manifests:")
        for path in hypersim_manifest_paths:
            print(f"  - {path}")
    else:
        hypersim_manifest = (
            args.hypersim_manifest
            or os.environ.get("HYPERSIM_MANIFEST_PATH")
            or data_cfg.get("hypersim_manifest_path")
        )
        if hypersim_manifest:
            resolved_manifest = os.path.expanduser(hypersim_manifest)
            data_cfg["hypersim_manifest_path"] = resolved_manifest
            data_cfg["hypersim_manifest_paths"] = [resolved_manifest]
            print(f"Using HyperSim manifest: {resolved_manifest}")

    hypersim_camera_meta = os.environ.get(
        "HYPERSIM_CAMERA_METADATA"
    ) or data_cfg.get("hypersim_camera_metadata_path")
    if hypersim_camera_meta:
        data_cfg["hypersim_camera_metadata_path"] = os.path.expanduser(
            hypersim_camera_meta
        )

    model_cfg = config.setdefault("model", {})
    pixel_encoder_cfg = model_cfg.setdefault("pixel_encoder", {})
    if args.fusion_layers:
        layers = []
        for item in args.fusion_layers.replace(",", " ").split():
            if item.strip():
                layers.append(int(item))
        if layers:
            pixel_encoder_cfg["fusion_layers"] = layers
    if args.use_cls_token:
        pixel_encoder_cfg["use_cls_token"] = True

    deterministic = config["generic"].get("deterministic", True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    # Disable mem-efficient SDPA: with fp16 + checkpointing the supposed memory
    # win does not materialise in practice.
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.set_num_threads(1)
    main_worker(config, args)
