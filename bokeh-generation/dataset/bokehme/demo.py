#!/usr/bin/env python
# encoding: utf-8
"""BokehMe inference pipeline used inside the synthetic data generation loop.

Only the ``pipeline`` function is exposed for production use. The original
CLI / file-based demo (image, depth, fg-mask reading + result saving) has
been removed from the public release since training/inference only need the
core renderer.
"""

import numpy as np
import torch
import torch.nn.functional as F


def gaussian_blur(x, r, sigma=None):
    """Pad-and-conv Gaussian blur used to feather the BokehMe error map."""
    r = int(round(r))
    if sigma is None:
        sigma = 0.3 * (r - 1) + 0.8
    x_grid, y_grid = torch.meshgrid(
        torch.arange(-int(r), int(r) + 1),
        torch.arange(-int(r), int(r) + 1),
        indexing="ij",
    )
    kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / 2 / sigma ** 2)
    kernel = kernel.float() / kernel.sum()
    kernel = kernel.expand(1, 1, 2 * r + 1, 2 * r + 1).to(x.device)
    x = F.pad(x, pad=(r, r, r, r), mode="replicate")
    x = F.conv2d(x, weight=kernel, padding=0)
    return x


def pipeline(classical_renderer, arnet, iunet, image, defocus, gamma, defocus_scale, gamma_min, gamma_max):
    """Run BokehMe end-to-end.

    Mixes a classical scatter pass with a neural refinement to produce the
    final bokeh image. ``classical_renderer`` is the ``ModuleRenderScatter``
    (or scatter_ex) instance; ``arnet`` / ``iunet`` are the neural blocks.
    """
    # Sanitize defocus numerics before using it for scaling
    defocus = torch.nan_to_num(defocus, nan=0.0, posinf=0.0, neginf=0.0)

    bokeh_classical, defocus_dilate = classical_renderer(image ** gamma, defocus * defocus_scale)

    bokeh_classical = bokeh_classical ** (1 / gamma)
    defocus_dilate = defocus_dilate / defocus_scale
    gamma = (gamma - gamma_min) / (gamma_max - gamma_min)
    adapt_scale = float(max(defocus.abs().amax().item(), 1.0))
    # Cap adapt_scale to image extent to avoid excessive loop/zero sizes
    H, W = image.shape[2], image.shape[3]
    adapt_scale = min(adapt_scale, float(max(H, W)))

    # Downsample using explicit size to avoid zero / overflow dimensions
    h0 = max(1, int(round(H / adapt_scale)))
    w0 = max(1, int(round(W / adapt_scale)))
    image_re = F.interpolate(image, size=(h0, w0), mode="bilinear", align_corners=True)
    defocus_re = (1.0 / adapt_scale) * F.interpolate(defocus, size=(h0, w0), mode="bilinear", align_corners=True)
    bokeh_neural, error_map = arnet(image_re, defocus_re, gamma)
    error_map = F.interpolate(error_map, size=(image.shape[2], image.shape[3]), mode="bilinear", align_corners=True)
    bokeh_neural.clamp_(0, 1e5)

    scale = -1
    for scale in range(int(np.log2(adapt_scale))):
        ratio = 2 ** (scale + 1) / adapt_scale
        h_re, w_re = int(ratio * image.shape[2]), int(ratio * image.shape[3])
        h_re = max(1, h_re)
        w_re = max(1, w_re)
        image_re = F.interpolate(image, size=(h_re, w_re), mode="bilinear", align_corners=True)
        defocus_re = ratio * F.interpolate(defocus, size=(h_re, w_re), mode="bilinear", align_corners=True)
        defocus_dilate_re = ratio * F.interpolate(defocus_dilate, size=(h_re, w_re), mode="bilinear", align_corners=True)
        bokeh_neural_refine = iunet(image_re, defocus_re.clamp(-1, 1), bokeh_neural, gamma).clamp(0, 1e5)
        mask = gaussian_blur(
            ((defocus_dilate_re < 1) * (defocus_dilate_re > -1)).float(),
            0.005 * (defocus_dilate_re.shape[2] + defocus_dilate_re.shape[3]),
        )
        bokeh_neural = mask * bokeh_neural_refine + (1 - mask) * F.interpolate(
            bokeh_neural, size=(h_re, w_re), mode="bilinear", align_corners=True
        )

    bokeh_neural_refine = iunet(image, defocus.clamp(-1, 1), bokeh_neural, gamma).clamp(0, 1e5)
    mask = gaussian_blur(
        ((defocus_dilate < 1) * (defocus_dilate > -1)).float(),
        0.005 * (defocus_dilate.shape[2] + defocus_dilate.shape[3]),
    )
    bokeh_neural = mask * bokeh_neural_refine + (1 - mask) * F.interpolate(
        bokeh_neural, size=(image.shape[2], image.shape[3]), mode="bilinear", align_corners=True
    )
    bokeh_pred = bokeh_classical * (1 - error_map) + bokeh_neural * error_map

    return bokeh_pred.clamp(0, 1), bokeh_classical.clamp(0, 1), bokeh_neural.clamp(0, 1), error_map
