"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import os

import torch
import torch.utils.data.distributed
import wandb
from torch.nn import functional as F

from unidepth.utils import barrier, is_main_process
from unidepth.utils.misc import remove_padding


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEBUG_LOG_ENABLED = _env_flag("UNIDEPTH_DEBUG")


def _debug_print(*args, **kwargs):
    if DEBUG_LOG_ENABLED:
        print(*args, **kwargs)


def original_image(batch, preds=None):
    paddings = [
        torch.tensor(pads)
        for img_meta in batch["img_metas"]
        for pads in img_meta.get("paddings", [[0] * 4])
    ]
    paddings = torch.stack(paddings).to(batch["data"]["image"].device)[
        ..., [0, 2, 1, 3]
    ]  # lrtb

    T, _, H, W = batch["data"]["depth"].shape
    batch["data"]["image"] = F.interpolate(
        batch["data"]["image"],
        (H + paddings[2] + paddings[3], W + paddings[1] + paddings[2]),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    batch["data"]["image"] = remove_padding(
        batch["data"]["image"], paddings.repeat(T, 1)
    )

    if preds is not None:
        for key in ["depth"]:
            if key in preds:
                preds[key] = F.interpolate(
                    preds[key],
                    (H + paddings[2] + paddings[3], W + paddings[1] + paddings[2]),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                preds[key] = remove_padding(preds[key], paddings.repeat(T, 1))

    return batch, preds


def log_metrics(metrics_all, step):
    for name_ds, metrics in metrics_all.items():
        for metrics_name, metrics_value in metrics.items():
            try:
                print(f"Metrics/{name_ds}/{metrics_name} {round(metrics_value, 4)}")
                wandb.log(
                    {f"Metrics/{name_ds}/{metrics_name}": metrics_value}, step=step
                )
            except:
                pass


def validate(
    model,
    test_loaders,
    step,
    context,
    visualize_fn=None,
    max_visualizations: int = 0,
):
    metrics_all = {}
    remaining_visualizations = max_visualizations
    samples_seen = 0
    for name_ds, test_loader in test_loaders.items():
        for i, batch in enumerate(test_loader):
            with context:
                batch["data"] = {
                    k: v.to(model.device) for k, v in batch["data"].items()
                }
                # === Debug: before squeeze ===
                if i == 0:
                    _debug_print(
                        f"[validation debug] Before squeeze, 'camera' in batch['data']: {'camera' in batch['data']}"
                    )
                    if "camera" in batch["data"]:
                        cam = batch["data"]["camera"]
                        _debug_print(f"[validation debug] Before squeeze, camera type: {type(cam)}")
                        _debug_print(f"[validation debug] Before squeeze, camera: {cam}")
                        if hasattr(cam, 'K'):
                            _debug_print(
                                f"[validation debug] Before squeeze, camera.K shape: {cam.K.shape}"
                            )

                # remove temporal dimension of the dataloder, here is always 1!
                batch["data"] = {k: v.squeeze(1) for k, v in batch["data"].items()}

                # === Debug: after squeeze ===
                if i == 0:
                    _debug_print(
                        f"[validation debug] After squeeze, 'camera' in batch['data']: {'camera' in batch['data']}"
                    )
                    if "camera" in batch["data"]:
                        cam = batch["data"]["camera"]
                        _debug_print(f"[validation debug] After squeeze, camera type: {type(cam)}")
                        if hasattr(cam, 'K'):
                            _debug_print(
                                f"[validation debug] After squeeze, camera.K shape: {cam.K.shape}"
                            )

                batch["img_metas"] = [
                    {k: v[0] for k, v in meta.items() if isinstance(v, list)}
                    for meta in batch["img_metas"]
                ]

                preds = model(batch["data"], batch["img_metas"])
                if i == 0 and is_main_process():
                    try:
                        depth_pred = preds.get("depth")
                        depth_gt = batch["data"].get("depth")
                        depth_mask = batch["data"].get("depth_mask")
                        if (
                            depth_pred is not None
                            and depth_gt is not None
                            and depth_mask is not None
                        ):
                            mask = depth_mask.bool()
                            valid_pred = depth_pred[mask]
                            valid_gt = depth_gt[mask]
                            if valid_pred.numel() > 0 and valid_gt.numel() > 0:
                                ratio = (
                                    valid_pred.median() / valid_gt.median()
                                ).item()
                                _debug_print(
                                    "[debug] batch0 depth stats: "
                                    f"pred_med={valid_pred.median().item():.4f}, "
                                    f"gt_med={valid_gt.median().item():.4f}, "
                                    f"ratio={ratio:.4f}"
                                )
                                _debug_print(
                                    "[debug] batch0 depth ranges: "
                                    f"pred[{valid_pred.min().item():.4f}, {valid_pred.max().item():.4f}], "
                                    f"gt[{valid_gt.min().item():.4f}, {valid_gt.max().item():.4f}]"
                                )
                        intrinsics = preds.get("intrinsics")
                        if intrinsics is not None:
                            sample = intrinsics.reshape(intrinsics.shape[0], -1)[0]
                            _debug_print(
                                "[debug] batch0 intrinsics sample:",
                                sample[:6].detach().cpu().tolist(),
                            )
                    except Exception as exc:
                        _debug_print(f"[debug] Failed to gather batch0 stats: {exc}")

            batch, _ = original_image(batch, preds=None)

            allow_all_ranks = bool(getattr(visualize_fn, "_allow_all_ranks", False))
            if (
                visualize_fn is not None
                and remaining_visualizations > 0
                and (allow_all_ranks or is_main_process())
            ):
                batch_size = batch["data"]["image"].shape[0]
                num_to_vis = min(batch_size, remaining_visualizations)
                for local_idx in range(num_to_vis):
                    try:
                        visualize_fn(
                            batch_inputs=batch["data"],
                            preds=preds,
                            sample_index=local_idx,
                            batch_index=i,
                            global_index=samples_seen + local_idx,
                            img_metas=batch["img_metas"],
                        )
                    except Exception as exc:
                        _debug_print(f"[validation] visualization callback failed: {exc}")
                remaining_visualizations -= num_to_vis

            test_loader.dataset.accumulate_metrics(
                inputs=batch["data"],
                preds=preds,
                keyframe_idx=batch["img_metas"][0].get("keyframe_idx"),
            )
            samples_seen += batch["data"]["image"].shape[0]
            if remaining_visualizations <= 0:
                visualize_fn = None

        barrier()
        metrics_all[name_ds] = test_loader.dataset.get_evaluation()

    barrier()
    if is_main_process():
        log_metrics(metrics_all=metrics_all, step=step)
    return metrics_all
