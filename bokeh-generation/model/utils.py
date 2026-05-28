import cv2
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F


def color_transfer_lab(source_pil, target_pil, adjust_std=True):
    source_np = np.array(source_pil, dtype=np.uint8)
    target_np = np.array(target_pil, dtype=np.uint8)
    source_lab = cv2.cvtColor(source_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    mean_src, std_src = cv2.meanStdDev(source_lab)
    mean_tgt, std_tgt = cv2.meanStdDev(target_lab)
    eps = 1e-6
    for c in range(3):
        t_chan = target_lab[..., c]
        t_chan -= mean_tgt[c][0]
        if adjust_std and std_tgt[c][0] > eps:
            t_chan *= (std_src[c][0] / std_tgt[c][0])
        t_chan += mean_src[c][0]
        target_lab[..., c] = t_chan
    target_lab = np.clip(target_lab, 0, 255).astype(np.uint8)
    matched = cv2.cvtColor(target_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(matched)

def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-6, mask=None):
    size = feat.size()
    if len(size) == 2:
        return calc_mean_std_2d(feat, eps, mask)
    assert (len(size) == 3)
    C = size[0]
    if mask is not None:
        feat_var = feat.view(C, -1)[:, mask.view(-1) == 1].var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1, 1)
        feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1].mean(dim=1).view(C, 1, 1)
    else:
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1, 1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1, 1)
    return feat_mean, feat_std

def calc_mean_std_2d(feat, eps=1e-6, mask=None):
    size = feat.size()
    assert (len(size) == 2)
    C = size[0]
    if mask is not None:
        feat_var = feat.view(C, -1)[:, mask.view(-1) == 1].var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1)
        feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1].mean(dim=1).view(C, 1)
    else:
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1)
    return feat_mean, feat_std


class BalancedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, foreground_token_attn_probs, foreground_masks):
        background_masks = 1 - foreground_masks
        background_masks_sum = background_masks.sum(dim=1) + 1e-5
        foreground_masks_sum = foreground_masks.sum(dim=1) + 1e-5

        background_loss = (foreground_token_attn_probs * background_masks).sum(dim=1) / background_masks_sum
        foreground_loss = (foreground_token_attn_probs * foreground_masks).sum(dim=1) / foreground_masks_sum

        return foreground_loss - background_loss

def get_attn_fgbg_loss_1_layer(cross_attention_scores, foreground_mask, loss_fn):
    b, _, num_noise_latents, _ = cross_attention_scores.shape
    size = int(num_noise_latents ** 0.5)

    foreground_mask = F.interpolate(foreground_mask, size=(size, size), mode="bilinear", antialias=True)  # (b, 1, size, size)
    foreground_mask = foreground_mask.view(b, -1)  # (b, size*size)

    cross_attention_scores = cross_attention_scores.squeeze(-1)  # (b, num_heads, num_noise_latents)
    cross_attention_scores = cross_attention_scores.mean(dim=1)  # (b, num_noise_latents)

    loss = loss_fn(cross_attention_scores, foreground_mask)

    return loss

def get_attn_loss(cross_attention_scores, loss_fn, label_map=None):
    num_layers = len(cross_attention_scores)
    loss = 0.0
    for _, v in cross_attention_scores.items():
        layer_loss = get_attn_fgbg_loss_1_layer(v, label_map, loss_fn)
        loss += layer_loss
    return loss / num_layers
