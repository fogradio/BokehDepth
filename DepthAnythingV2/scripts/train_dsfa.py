#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader


RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from datasets import DefocusStackDataset, collate_defocus_stack  # noqa: E402
from depth_anything_v2 import MODEL_CONFIGS, DepthAnythingV2DSFA  # noqa: E402


LOGGER = logging.getLogger("depthanythingv2_dsfa_train")


class SiLogLoss(nn.Module):
    def __init__(self, lambd: float = 0.5) -> None:
        super().__init__()
        self.lambd = float(lambd)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        pred = pred.squeeze()
        target = target.squeeze()
        valid_mask = valid_mask.squeeze().bool()
        if valid_mask.shape != pred.shape:
            valid_mask = valid_mask.expand_as(pred)

        eps = 1e-6
        valid_mask = (
            valid_mask
            & torch.isfinite(pred)
            & torch.isfinite(target)
            & (pred > eps)
            & (target > eps)
        )
        if valid_mask.sum() < 10:
            return pred.sum() * 0.0

        mask = valid_mask.float()
        pred_safe = torch.clamp(pred, min=eps)
        target_safe = torch.clamp(target, min=eps)
        diff_log = (torch.log(target_safe) - torch.log(pred_safe)) * mask
        valid_count = mask.sum().clamp_min(1.0)
        diff_mean = diff_log.sum() / valid_count
        diff_sq_mean = (diff_log.square()).sum() / valid_count
        return torch.sqrt(torch.clamp(diff_sq_mean - self.lambd * diff_mean.square(), min=0.0))


def load_checkpoint_state(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    return {
        key.removeprefix("module."): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }


def load_json_config(path: Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def default_from(config: dict, key: str, fallback):
    return config.get(key, fallback)


def build_parser(defaults: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Depth Anything V2 with DSFA on manifest-backed defocus stacks."
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON defaults file")
    parser.add_argument("--manifest-path", type=Path, default=default_from(defaults, "manifest_path", None))
    parser.add_argument("--save-path", type=Path, default=default_from(defaults, "save_path", None))
    parser.add_argument("--pretrained-from", type=Path, default=default_from(defaults, "pretrained_from", None))
    parser.add_argument("--resume-from", type=Path, default=default_from(defaults, "resume_from", None))
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default=default_from(defaults, "encoder", "vitl"))
    parser.add_argument("--img-size", type=int, default=default_from(defaults, "img_size", 518))
    parser.add_argument("--min-depth", type=float, default=default_from(defaults, "min_depth", 0.001))
    parser.add_argument("--max-depth", type=float, default=default_from(defaults, "max_depth", 80.0))
    parser.add_argument("--depth-scale", type=float, default=default_from(defaults, "depth_scale", 1.0))
    parser.add_argument("--epochs", type=int, default=default_from(defaults, "epochs", 40))
    parser.add_argument("--batch-size", type=int, default=default_from(defaults, "batch_size", 2))
    parser.add_argument("--num-workers", type=int, default=default_from(defaults, "num_workers", 4))
    parser.add_argument("--lr", type=float, default=default_from(defaults, "lr", 5e-6))
    parser.add_argument("--weight-decay", type=float, default=default_from(defaults, "weight_decay", 0.01))
    parser.add_argument("--grad-accum-steps", type=int, default=default_from(defaults, "grad_accum_steps", 1))
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default=default_from(defaults, "mixed_precision", "no"))
    parser.add_argument("--stack-indices", nargs="+", type=int, default=default_from(defaults, "stack_indices", None))
    parser.add_argument("--fusion-layers", nargs="+", type=int, default=default_from(defaults, "fusion_layers", [2, 3]))
    parser.add_argument("--num-heads", type=int, default=default_from(defaults, "num_heads", 8))
    parser.add_argument("--attn-dropout", type=float, default=default_from(defaults, "attn_dropout", 0.0))
    parser.add_argument("--layerscale-init", type=float, default=default_from(defaults, "layerscale_init", 0.1))
    parser.add_argument("--alibi-scale", type=float, default=default_from(defaults, "alibi_scale", 1.0))
    parser.add_argument("--use-sdpa", action="store_true", default=default_from(defaults, "use_sdpa", False))
    parser.add_argument("--grad-free-stack", action="store_true", default=default_from(defaults, "grad_free_stack", False))
    parser.add_argument("--save-every", type=int, default=default_from(defaults, "save_every", 1))
    parser.add_argument("--log-every", type=int, default=default_from(defaults, "log_every", 10))
    return parser


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, _ = config_parser.parse_known_args()
    defaults = load_json_config(config_args.config)
    parser = build_parser(defaults)
    args = parser.parse_args()
    if not args.manifest_path:
        parser.error("--manifest-path is required")
    if not args.save_path:
        parser.error("--save-path is required")
    if not args.pretrained_from:
        args.pretrained_from = None
    if not args.resume_from:
        args.resume_from = None
    return args


def configure_logging(is_main_process: bool) -> None:
    level = logging.INFO if is_main_process else logging.ERROR
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_model(args: argparse.Namespace) -> DepthAnythingV2DSFA:
    return DepthAnythingV2DSFA(
        **MODEL_CONFIGS[args.encoder],
        max_depth=args.max_depth,
        fusion_layers=args.fusion_layers,
        grad_free_stack=args.grad_free_stack,
        num_heads=args.num_heads,
        attn_dropout=args.attn_dropout,
        layerscale_init=args.layerscale_init,
        alibi_scale=args.alibi_scale,
        use_sdpa=args.use_sdpa,
    )


def maybe_load_weights(model: nn.Module, args: argparse.Namespace) -> int:
    load_path = args.resume_from or args.pretrained_from
    if load_path is None:
        return 0

    state = load_checkpoint_state(load_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    LOGGER.info("Loaded weights from %s", load_path)
    LOGGER.info("Missing keys: %d, unexpected keys: %d", len(missing), len(unexpected))

    if args.resume_from is None:
        return 0

    checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
    return int(checkpoint.get("epoch", -1)) + 1


def save_checkpoint(
    accelerator: Accelerator,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    save_path: Path,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": accelerator.get_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
    }
    accelerator.save(checkpoint, save_path / f"epoch_{epoch + 1}.pth")
    accelerator.save(checkpoint, save_path / "latest.pth")


def main() -> None:
    args = parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum_steps,
    )
    configure_logging(accelerator.is_main_process)

    if args.use_sdpa and torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    dataset = DefocusStackDataset(
        manifest_path=args.manifest_path,
        mode="train",
        size=(args.img_size, args.img_size),
        stack_indices=args.stack_indices,
        depth_scale=args.depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_defocus_stack,
    )

    model = build_model(args)
    start_epoch = maybe_load_weights(model, args)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = SiLogLoss()

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

    if accelerator.is_main_process:
        LOGGER.info("Training samples: %d", len(dataset))
        LOGGER.info("Starting at epoch %d / %d", start_epoch, args.epochs)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(loader):
            with accelerator.accumulate(model):
                pred = model(batch["image"], batch["focus_stack"], batch["k_stack"])
                loss = criterion(pred, batch["depth"], batch["valid_mask"])
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += float(loss.detach().float().cpu())
            if accelerator.is_main_process and step % max(1, args.log_every) == 0:
                LOGGER.info(
                    "epoch=%d step=%d/%d loss=%.5f",
                    epoch,
                    step,
                    len(loader),
                    running_loss / max(1, step + 1),
                )

        if accelerator.is_main_process and (
            (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs
        ):
            save_checkpoint(accelerator, model, optimizer, args.save_path, epoch, args)
            LOGGER.info("Saved checkpoint for epoch %d", epoch + 1)

    if accelerator.is_main_process:
        save_checkpoint(accelerator, model, optimizer, args.save_path, args.epochs - 1, args)
        LOGGER.info("Training finished")


if __name__ == "__main__":
    main()
