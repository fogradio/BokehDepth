from collections import defaultdict
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F

from unidepth.utils.chamfer_distance import ChamferDistance

chamfer_cls = ChamferDistance()
EPS = 1e-6


def _clamp_positive(tensor: torch.Tensor, max_value: Optional[float] = None):
    if max_value is not None:
        return torch.clamp(tensor, min=EPS, max=max_value)
    return torch.clamp(tensor, min=EPS)


def chamfer_dist(tensor1, tensor2):
    x_lengths = torch.tensor((tensor1.shape[1],), device=tensor1.device)
    y_lengths = torch.tensor((tensor2.shape[1],), device=tensor2.device)
    dist1, dist2, idx1, idx2 = chamfer_cls(
        tensor1, tensor2, x_lengths=x_lengths, y_lengths=y_lengths
    )
    return (torch.sqrt(dist1) + torch.sqrt(dist2)) / 2


def auc(tensor1, tensor2, thresholds):
    x_lengths = torch.tensor((tensor1.shape[1],), device=tensor1.device)
    y_lengths = torch.tensor((tensor2.shape[1],), device=tensor2.device)
    dist1, dist2, idx1, idx2 = chamfer_cls(
        tensor1, tensor2, x_lengths=x_lengths, y_lengths=y_lengths
    )
    # compute precision recall
    precisions = [(dist1 < threshold).sum() / dist1.numel() for threshold in thresholds]
    recalls = [(dist2 < threshold).sum() / dist2.numel() for threshold in thresholds]
    auc_value = torch.trapz(
        torch.tensor(precisions, device=tensor1.device),
        torch.tensor(recalls, device=tensor1.device),
    )
    return auc_value


def _build_fa_thresholds(thresholds: torch.Tensor) -> torch.Tensor:
    if thresholds is None or thresholds.numel() == 0:
        raise ValueError("FA metric requires a non-empty thresholds tensor.")
    max_tau = torch.max(thresholds)
    steps = int(max(thresholds.numel(), 2))
    if max_tau <= 0:
        return torch.zeros(
            steps,
            device=thresholds.device,
            dtype=thresholds.dtype,
        )
    return torch.linspace(
        0.0,
        max_tau,
        steps=steps,
        device=thresholds.device,
        dtype=thresholds.dtype,
    )


def _pairwise_nn_distances(
    pred_points: torch.Tensor, gt_points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_batch = pred_points.t().unsqueeze(0)  # (1, N_pred, 3)
    gt_batch = gt_points.t().unsqueeze(0)  # (1, N_gt, 3)
    x_lengths = torch.tensor((pred_batch.shape[1],), device=pred_points.device)
    y_lengths = torch.tensor((gt_batch.shape[1],), device=gt_points.device)
    dist_pred_to_gt, dist_gt_to_pred, _, _ = chamfer_cls(
        pred_batch, gt_batch, x_lengths=x_lengths, y_lengths=y_lengths
    )
    return torch.sqrt(dist_pred_to_gt.squeeze(0)), torch.sqrt(
        dist_gt_to_pred.squeeze(0)
    )


def _fa_auc_single(
    pred_points: torch.Tensor, gt_points: torch.Tensor, thresholds: torch.Tensor
) -> torch.Tensor:
    if pred_points.numel() == 0 or gt_points.numel() == 0:
        return torch.zeros(
            1, device=pred_points.device, dtype=pred_points.dtype
        ).squeeze()
    thresholds = torch.sort(torch.clamp(thresholds, min=0.0))[0]
    dist_pred_to_gt, dist_gt_to_pred = _pairwise_nn_distances(pred_points, gt_points)
    if dist_pred_to_gt.numel() == 0 or dist_gt_to_pred.numel() == 0:
        return torch.zeros(
            1, device=pred_points.device, dtype=pred_points.dtype
        ).squeeze()
    thresholds_expanded = thresholds.view(1, -1)
    precisions = (
        (dist_pred_to_gt.unsqueeze(1) <= thresholds_expanded).to(torch.float32).mean(0)
    )
    recalls = (
        (dist_gt_to_pred.unsqueeze(1) <= thresholds_expanded).to(torch.float32).mean(0)
    )
    f1_curve = 2 * precisions * recalls / (precisions + recalls + EPS)
    denom = thresholds[-1] - thresholds[0]
    if denom <= EPS:
        return torch.zeros(
            1, device=pred_points.device, dtype=pred_points.dtype
        ).squeeze()
    fa_value = torch.trapz(f1_curve, thresholds) / denom
    return fa_value.clamp(min=0.0, max=1.0)


def _align_pred_points(
    pred_points: torch.Tensor, gt_points: torch.Tensor, mode: str
) -> torch.Tensor:
    if mode == "none":
        return pred_points
    if mode not in {"si", "ssi"}:
        raise ValueError(f"Unsupported FA alignment mode: {mode}")

    gt_depth = _clamp_positive(gt_points[2])
    pred_depth = _clamp_positive(pred_points[2])

    ray_directions = pred_points / pred_points[2].clamp(min=EPS).unsqueeze(0)
    if mode == "si":
        aligned_depth = si(gt_depth, pred_depth)
    else:
        aligned_depth = ssi(gt_depth, pred_depth)
    aligned_depth = _clamp_positive(aligned_depth)
    return ray_directions * aligned_depth.unsqueeze(0)


def _fa_wrapper(mode: str):
    def _compute(gt_points: torch.Tensor, pred_points: torch.Tensor, thresholds):
        fa_thresholds = _build_fa_thresholds(thresholds)
        aligned_pred = _align_pred_points(pred_points, gt_points, mode)
        fa_score = _fa_auc_single(aligned_pred, gt_points, fa_thresholds)
        return torch.atleast_1d(fa_score)

    return _compute


def _f1_wrapper(mode: str):
    """Wrapper around the F1 metric, supporting si and ssi alignment modes."""
    def _compute(gt_points: torch.Tensor, pred_points: torch.Tensor, thresholds):
        aligned_pred = _align_pred_points(pred_points, gt_points, mode)
        f1_val = f1_score(
            gt_points.unsqueeze(0).permute(0, 2, 1),
            aligned_pred.unsqueeze(0).permute(0, 2, 1),
            thresholds=thresholds,
        )
        return f1_val

    return _compute


def delta(tensor1, tensor2, exponent):
    inlier = torch.maximum((tensor1 / tensor2), (tensor2 / tensor1))
    return (inlier < 1.25**exponent).to(torch.float32).mean()


def tau(tensor1, tensor2, perc):
    inlier = torch.maximum((tensor1 / tensor2), (tensor2 / tensor1))
    return (inlier < (1.0 + perc)).to(torch.float32).mean()


def ssi(tensor1, tensor2):
    tensor1 = _clamp_positive(tensor1)
    tensor2 = _clamp_positive(tensor2)
    if tensor1.numel() < 2:
        return si(tensor1, tensor2)

    stability_mat = 1e-9 * torch.eye(2, device=tensor1.device, dtype=tensor1.dtype)
    tensor2_one = torch.stack(
        [tensor2.detach(), torch.ones_like(tensor2).detach()], dim=1
    )
    lhs = tensor2_one.T @ tensor2_one + stability_mat
    rhs = tensor2_one.T @ tensor1.unsqueeze(1)

    try:
        scale_shift = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        # fall back to pseudo-inverse when the system is close to singular
        scale_shift = torch.linalg.pinv(lhs) @ rhs

    scale_shift = scale_shift.squeeze()
    if not torch.isfinite(scale_shift).all():
        return si(tensor1, tensor2)

    scale, shift = scale_shift.chunk(2, dim=0)
    return tensor2 * scale + shift


def si(tensor1, tensor2):
    tensor1 = _clamp_positive(tensor1)
    tensor2 = _clamp_positive(tensor2)
    median_den = torch.median(tensor2)
    if median_den.abs() < EPS:
        return tensor2
    return tensor2 * torch.median(tensor1) / median_den


def arel(tensor1, tensor2):
    tensor1 = _clamp_positive(tensor1)
    tensor2 = _clamp_positive(tensor2)
    median_den = torch.median(tensor2)
    if median_den.abs() < EPS:
        return torch.zeros(1, device=tensor1.device, dtype=tensor1.dtype)
    tensor2 = tensor2 * torch.median(tensor1) / median_den
    return (torch.abs(tensor1 - tensor2) / tensor1).mean()


RESCALE_METRICS = {
    "d1",
    "d2",
    "d3",
    "tau",
    "arel",
    "sqrel",
    "rmse",
    "rmselog",
    "log10",
    "silog",
}
RESCALE_FUNCS = {
    "ssi": ssi,
    "si": si,
}


def d_auc(tensor1, tensor2):
    exponents = torch.linspace(0.01, 5.0, steps=100, device=tensor1.device)
    deltas = [delta(tensor1, tensor2, exponent) for exponent in exponents]
    return torch.trapz(torch.tensor(deltas, device=tensor1.device), exponents) / 5.0


def f1_score(tensor1, tensor2, thresholds):
    x_lengths = torch.tensor((tensor1.shape[1],), device=tensor1.device)
    y_lengths = torch.tensor((tensor2.shape[1],), device=tensor2.device)
    dist1, dist2, idx1, idx2 = chamfer_cls(
        tensor1, tensor2, x_lengths=x_lengths, y_lengths=y_lengths
    )
    # compute precision recall
    precisions = [(dist1 < threshold).sum() / dist1.numel() for threshold in thresholds]
    recalls = [(dist2 < threshold).sum() / dist2.numel() for threshold in thresholds]
    precisions = torch.tensor(precisions, device=tensor1.device)
    recalls = torch.tensor(recalls, device=tensor1.device)
    f1_thresholds = 2 * precisions * recalls / (precisions + recalls)
    f1_thresholds = torch.where(
        torch.isnan(f1_thresholds), torch.zeros_like(f1_thresholds), f1_thresholds
    )
    f1_value = torch.trapz(f1_thresholds) / len(thresholds)
    return f1_value


DICT_METRICS = {
    "d1": partial(delta, exponent=1.0),
    "d2": partial(delta, exponent=2.0),
    "d3": partial(delta, exponent=3.0),
    "rmse": lambda gt, pred: torch.sqrt(((gt - pred) ** 2).mean()),
    "rmselog": lambda gt, pred: torch.sqrt(
        ((torch.log(gt) - torch.log(pred)) ** 2).mean()
    ),
    "arel": lambda gt, pred: (torch.abs(gt - pred) / gt).mean(),
    "sqrel": lambda gt, pred: (((gt - pred) ** 2) / gt).mean(),
    "log10": lambda gt, pred: torch.abs(torch.log10(pred) - torch.log10(gt)).mean(),
    "silog": lambda gt, pred: 100 * torch.std(torch.log(pred) - torch.log(gt)).mean(),
    "medianlog": lambda gt, pred: 100
    * (torch.log(pred) - torch.log(gt)).median().abs(),
    "d_auc": d_auc,
    "tau": partial(tau, perc=0.03),
}


DICT_METRICS_3D = {
    "MSE_3d": lambda gt, pred, thresholds: torch.norm(gt - pred, dim=0, p=2),
    "chamfer": lambda gt, pred, thresholds: chamfer_dist(
        gt.unsqueeze(0).permute(0, 2, 1), pred.unsqueeze(0).permute(0, 2, 1)
    ),
    "F1": lambda gt, pred, thresholds: f1_score(
        gt.unsqueeze(0).permute(0, 2, 1),
        pred.unsqueeze(0).permute(0, 2, 1),
        thresholds=thresholds,
    ),
    "F1_si": _f1_wrapper("si"),
    "F1_ssi": _f1_wrapper("ssi"),
    "FA": _fa_wrapper("none"),
    "FA_si": _fa_wrapper("si"),
    "FA_ssi": _fa_wrapper("ssi"),
}

DICT_METRICS_D = {
    "a1": lambda gt, pred: (torch.maximum((gt / pred), (pred / gt)) > 1.25**1.0).to(
        torch.float32
    ),
    "abs_rel": lambda gt, pred: (torch.abs(gt - pred) / gt),
}


def eval_depth(
    gts: torch.Tensor, preds: torch.Tensor, masks: torch.Tensor, max_depth=None
):
    def _append_zero(metric_dict, suffixes):
        zero_val = torch.zeros(1, device=gts.device, dtype=gts.dtype)
        for metric_name in suffixes:
            metric_dict[metric_name].append(zero_val)

    summary_metrics = defaultdict(list)
    preds = F.interpolate(preds, gts.shape[-2:], mode="bilinear")
    for i, (gt, pred, mask) in enumerate(zip(gts, preds, masks)):
        mask = mask.bool() & torch.isfinite(gt) & torch.isfinite(pred)
        mask = mask & (gt > EPS) & (pred > EPS)
        if max_depth is not None:
            mask = mask & (gt <= max_depth)
        if not torch.any(mask):
            metric_keys = list(DICT_METRICS.keys())
            extra_keys = []
            for name in metric_keys:
                if name in RESCALE_METRICS:
                    extra_keys.extend([f"{name}_ssi", f"{name}_si"])
            # ensure base keys present in dict before appending
            for base in metric_keys + extra_keys:
                summary_metrics.setdefault(base, [])
            _append_zero(summary_metrics, metric_keys + extra_keys)
            continue

        gt_vals = _clamp_positive(gt[mask], max_depth)
        pred_vals = _clamp_positive(pred[mask], max_depth)

        for name, fn in DICT_METRICS.items():
            metric_score = torch.atleast_1d(fn(gt_vals, pred_vals))
            summary_metrics[name].append(metric_score)

            if name in RESCALE_METRICS:
                for rescale_name, rescale_fn in RESCALE_FUNCS.items():
                    rescaled_pred = _clamp_positive(rescale_fn(gt_vals, pred_vals), max_depth)
                    metric_val = torch.atleast_1d(fn(gt_vals, rescaled_pred))
                    summary_metrics[f"{name}_{rescale_name}"].append(metric_val)
    return {name: torch.stack(vals, dim=0) for name, vals in summary_metrics.items()}


def eval_3d(
    gts: torch.Tensor, preds: torch.Tensor, masks: torch.Tensor, thresholds=None
):
    summary_metrics = defaultdict(list)
    ratio = min(
        1.0, (240 * 320 / masks.sum()) ** 0.5
    )  # rescale to avoid OOM during eval, FIXME
    h_max, w_max = int(gts.shape[-2] * ratio), int(gts.shape[-1] * ratio)
    gts = F.interpolate(gts, size=(h_max, w_max), mode="nearest-exact")
    preds = F.interpolate(preds, size=(h_max, w_max), mode="nearest-exact")
    masks = F.interpolate(
        masks.float(), size=(h_max, w_max), mode="nearest-exact"
    ).bool()
    for i, (gt, pred, mask) in enumerate(zip(gts, preds, masks)):
        if not torch.any(mask):
            continue
        finite_mask = mask & torch.isfinite(gt).all(dim=0, keepdim=True)
        finite_mask = finite_mask & torch.isfinite(pred).all(dim=0, keepdim=True)
        if not torch.any(finite_mask):
            continue
        for name, fn in DICT_METRICS_3D.items():
            metric_val = fn(
                gt[:, finite_mask.squeeze()],
                pred[:, finite_mask.squeeze()],
                thresholds,
            ).mean()
            summary_metrics[name].append(metric_val)

    metrics_dict = {
        name: torch.stack(vals, dim=0) for name, vals in summary_metrics.items()
    }

    # Print FA metrics.
    fa_print_sequence = [("FA", "none"), ("FA_si", "si"), ("FA_ssi", "ssi")]
    fa_labels = []
    fa_values = []
    for key, label in fa_print_sequence:
        if key in metrics_dict and metrics_dict[key].numel() > 0:
            fa_labels.append(label)
            fa_values.append(metrics_dict[key].mean().detach().cpu().item() * 100.0)
    if fa_values:
        label_str = "/".join(fa_labels)
        value_str = " / ".join(f"{val:.2f}%" for val in fa_values)
        print(f"[eval_3d] FA ({label_str}): {value_str}")

    # Print F1 metrics.
    f1_print_sequence = [("F1", "none"), ("F1_si", "si"), ("F1_ssi", "ssi")]
    f1_labels = []
    f1_values = []
    for key, label in f1_print_sequence:
        if key in metrics_dict and metrics_dict[key].numel() > 0:
            f1_labels.append(label)
            f1_values.append(metrics_dict[key].mean().detach().cpu().item() * 100.0)
    if f1_values:
        label_str = "/".join(f1_labels)
        value_str = " / ".join(f"{val:.2f}%" for val in f1_values)
        print(f"[eval_3d] F1 ({label_str}): {value_str}")

    return metrics_dict
