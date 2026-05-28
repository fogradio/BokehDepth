"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from PIL import Image

from unidepth.utils.misc import ssi_helper


def colorize(
    value: np.ndarray, vmin: float = None, vmax: float = None, cmap: str = "magma_r"
):
    # if already RGB, do nothing
    if value.ndim > 2:
        if value.shape[-1] > 1:
            return value
        value = value[..., 0]
    invalid_mask = value < 0.0001
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    denom = vmax - vmin
    if abs(denom) < 1e-9:
        denom = 1.0
    value = (value - vmin) / denom  # vmin..vmax

    # set color
    cmapper = plt.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 0
    img = value[..., :3]
    return img


def image_grid(imgs: list[np.ndarray], rows: int, cols: int) -> np.ndarray:
    if not len(imgs):
        return None
    assert len(imgs) == rows * cols
    h, w = imgs[0].shape[:2]
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(
            Image.fromarray(img.astype(np.uint8)).resize(
                (w, h), resample=Image.BILINEAR
            ),
            box=(i % cols * w, i // cols * h),
        )

    return np.array(grid)


def get_pointcloud_from_rgbd(
    image: np.array,
    depth: np.array,
    mask: np.ndarray,
    intrinsic_matrix: np.array,
    extrinsic_matrix: np.array = None,
):
    depth = np.array(depth).squeeze()
    mask = np.array(mask).squeeze()
    # Mask the depth array
    masked_depth = np.ma.masked_where(mask == False, depth)
    # masked_depth = np.ma.masked_greater(masked_depth, 8000)
    # Create idx array
    idxs = np.indices(masked_depth.shape)
    u_idxs = idxs[1]
    v_idxs = idxs[0]
    # Get only non-masked depth and idxs
    z = masked_depth[~masked_depth.mask]
    compressed_u_idxs = u_idxs[~masked_depth.mask]
    compressed_v_idxs = v_idxs[~masked_depth.mask]
    image = np.stack(
        [image[..., i][~masked_depth.mask] for i in range(image.shape[-1])], axis=-1
    )

    # Calculate local position of each point
    # Apply vectorized math to depth using compressed arrays
    cx = intrinsic_matrix[0, 2]
    fx = intrinsic_matrix[0, 0]
    x = (compressed_u_idxs - cx) * z / fx
    cy = intrinsic_matrix[1, 2]
    fy = intrinsic_matrix[1, 1]
    # Flip y as we want +y pointing up not down
    y = -((compressed_v_idxs - cy) * z / fy)

    # # Apply camera_matrix to pointcloud as to get the pointcloud in world coords
    # if extrinsic_matrix is not None:
    #     # Calculate camera pose from extrinsic matrix
    #     camera_matrix = np.linalg.inv(extrinsic_matrix)
    #     # Create homogenous array of vectors by adding 4th entry of 1
    #     # At the same time flip z as for eye space the camera is looking down the -z axis
    #     w = np.ones(z.shape)
    #     x_y_z_eye_hom = np.vstack((x, y, -z, w))
    #     # Transform the points from eye space to world space
    #     x_y_z_world = np.dot(camera_matrix, x_y_z_eye_hom)[:3]
    #     return x_y_z_world.T
    # else:
    x_y_z_local = np.stack((x, y, z), axis=-1)
    return np.concatenate([x_y_z_local, image], axis=-1)


def save_file_ply(xyz, rgb, pc_file):
    if rgb.max() < 1.001:
        rgb = rgb * 255.0
    rgb = rgb.astype(np.uint8)
    # print(rgb)
    with open(pc_file, "w") as f:
        # headers
        f.writelines(
            [
                "ply\n" "format ascii 1.0\n",
                "element vertex {}\n".format(xyz.shape[0]),
                "property float x\n",
                "property float y\n",
                "property float z\n",
                "property uchar red\n",
                "property uchar green\n",
                "property uchar blue\n",
                "end_header\n",
            ]
        )

        for i in range(xyz.shape[0]):
            str_v = "{:10.6f} {:10.6f} {:10.6f} {:d} {:d} {:d}\n".format(
                xyz[i, 0], xyz[i, 1], xyz[i, 2], rgb[i, 0], rgb[i, 1], rgb[i, 2]
            )
            f.write(str_v)


# really awful fct... FIXME
def log_train_artifacts(rgbs, gts, preds, ds_name, step, infos={}):
    rgbs = [
        (127.5 * (rgb + 1))
        .clip(0, 255)
        .to(torch.uint8)
        .cpu()
        .detach()
        .permute(1, 2, 0)
        .numpy()
        for rgb in rgbs
    ]

    new_gts, new_preds = [], []
    if len(gts) > 0:
        for i, gt in enumerate(gts):
            scale, shift = ssi_helper(
                gts[i][gts[i] > 0].cpu().detach(), preds[i][gts[i] > 0].cpu().detach()
            )
            gt = gts[i].cpu().detach().squeeze().numpy()
            pred = (preds[i].cpu().detach() * scale + shift).squeeze().numpy()
            vmin = gt[gt > 0].min() if (gt > 0).any() else 0.0
            vmax = gt.max() if (gt > 0).any() else 0.1
            new_gts.append(colorize(gt, vmin=vmin, vmax=vmax))
            new_preds.append(colorize(pred, vmin=vmin, vmax=vmax))
        gts, preds = new_gts, new_preds
    else:
        preds = [
            colorize(pred.cpu().detach().squeeze().numpy(), 0.0, 80.0)
            for i, pred in enumerate(preds)
        ]

    num_additional, additionals = 0, []
    for name, info in infos.items():
        num_additional += 1
        if info.shape[1] == 3:
            additionals.extend(
                [
                    (127.5 * (x + 1))
                    .clip(0, 255)
                    .to(torch.uint8)
                    .cpu()
                    .detach()
                    .permute(1, 2, 0)
                    .numpy()
                    for x in info[:4]
                ]
            )
        else:
            additionals.extend(
                [
                    colorize(x.cpu().detach().squeeze().numpy())
                    for i, x in enumerate(info[:4])
                ]
            )

    num_rows = 2 + int(len(gts) > 0) + num_additional
    artifacts_grid = image_grid(
        [*rgbs, *gts, *preds, *additionals], num_rows, len(rgbs)
    )
    try:
        wandb.log({f"{ds_name}_training": [wandb.Image(artifacts_grid)]}, step=step)
    except:
        Image.fromarray(artifacts_grid).save(
            os.path.join(os.environ["HOME"], "Workspace", f"art_grid{step}.png")
        )


def _to_numpy(array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    return np.asarray(array)


def save_eval_visualization(
    rgb: Union[np.ndarray, torch.Tensor],
    pred_depth: Union[np.ndarray, torch.Tensor],
    gt_depth: Optional[Union[np.ndarray, torch.Tensor]] = None,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    out_path: Optional[Union[str, Path]] = None,
    *,
    align: bool = True,
    cmap: str = "magma_r",
    save_pred_color: bool = True,
    save_pred_depth: bool = False,
) -> np.ndarray:
    """Render an evaluation triplet (RGB, prediction, GT) and optionally save it.

    Args:
        rgb: RGB image
        pred_depth: predicted depth map
        gt_depth: ground-truth depth map (optional)
        mask: evaluation mask (optional)
        out_path: path for the triplet image (optional)
        align: whether to align the prediction to GT via scale/shift
        cmap: depth visualization colormap
        save_pred_color: also save the colored prediction (middle panel) on its own
        save_pred_depth: also save the raw predicted depth (.npy)

    Returns:
        numpy array of the triplet image
    """

    rgb_np = _to_numpy(rgb)
    if rgb_np.ndim == 3 and rgb_np.shape[0] in {1, 3}:
        rgb_np = np.transpose(rgb_np, (1, 2, 0))
    rgb_np = np.clip(rgb_np, 0.0, 1.0)
    rgb_img = (rgb_np * 255.0).astype(np.uint8)

    pred_np = _to_numpy(pred_depth).squeeze()
    gt_np = _to_numpy(gt_depth).squeeze() if gt_depth is not None else None
    mask_np = _to_numpy(mask).squeeze() if mask is not None else None

    if mask_np is None:
        if gt_np is not None:
            mask_np = np.isfinite(gt_np) & (gt_np > 0)
        else:
            mask_np = np.isfinite(pred_np)

    pred_vis = pred_np.copy()
    if gt_np is not None and align:
        valid = mask_np & np.isfinite(gt_np) & np.isfinite(pred_np)
        if np.count_nonzero(valid) > 0:
            gt_valid = torch.from_numpy(gt_np[valid])
            pred_valid = torch.from_numpy(pred_np[valid])
            scale, shift = ssi_helper(gt_valid, pred_valid)
            pred_vis = pred_np * float(scale) + float(shift)

    if gt_np is not None:
        valid_gt = mask_np & np.isfinite(gt_np) & (gt_np > 0)
        if np.count_nonzero(valid_gt) > 0:
            vmin = float(gt_np[valid_gt].min())
            vmax = float(gt_np[valid_gt].max())
        else:
            vmin, vmax = float(gt_np.min()), float(gt_np.max())
    else:
        valid_pred = mask_np & np.isfinite(pred_vis)
        vmin = float(pred_vis[valid_pred].min()) if np.any(valid_pred) else float(pred_vis.min())
        vmax = float(pred_vis[valid_pred].max()) if np.any(valid_pred) else float(pred_vis.max())

    pred_img = colorize(pred_vis, vmin=vmin, vmax=vmax, cmap=cmap)

    panels = [rgb_img, pred_img]
    if gt_np is not None:
        gt_img = colorize(gt_np, vmin=vmin, vmax=vmax, cmap=cmap)
        panels.append(gt_img)

    grid = image_grid(panels, 1, len(panels))
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Save the triplet image.
        Image.fromarray(grid).save(out_path)

        # Save the colored prediction on its own (middle panel of the triplet).
        if save_pred_color:
            pred_color_path = out_path.parent / f"{out_path.stem}_pred_color{out_path.suffix}"
            Image.fromarray(pred_img).save(pred_color_path)

        # Save the raw predicted depth values.
        if save_pred_depth:
            pred_depth_path = out_path.parent / f"{out_path.stem}_pred_depth.npy"
            np.save(pred_depth_path, pred_np)

    return grid
