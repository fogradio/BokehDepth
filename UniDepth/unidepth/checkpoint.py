"""UniDepth config and checkpoint loading utilities.

Contains only the two functions actually used by the inference pipeline:
    - load_config: read a UniDepth JSON config file.
    - instantiate_model: build the model from config and load regular / EMA
      weights, automatically stripping the "module." prefix left by DDP.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import torch


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Read a UniDepth JSON config file and return it as a dict."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_model_class(config: Dict[str, Any]):
    """Look up the model class in unidepth.models by config["model"]["name"].

    The import is deferred to avoid a circular import if unidepth.checkpoint
    is pulled in transitively during unidepth.models initialization.
    """
    import unidepth.models as model_zoo

    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "UniDepthV2")
    try:
        return getattr(model_zoo, model_name)
    except AttributeError as exc:
        raise ValueError(
            f"Model '{model_name}' is not available in unidepth.models"
        ) from exc


def instantiate_model(
    config: Dict[str, Any],
    weights_path: Union[str, Path],
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate the model and load its checkpoint.

    Supports two save formats: the training-time {"model": state_dict, "ema": ...}
    layout, and a bare state_dict. The "module." prefix left by DDP wrapping is
    stripped automatically; if the checkpoint carries EMA parameters they are
    loaded on top (inference typically prefers EMA).
    """
    weights_path = Path(weights_path)
    model_cls = _resolve_model_class(config)
    model = model_cls(config).to(device)

    # Load checkpoint: weights_only only exists on newer torch; fall back for older versions.
    load_kwargs = {"map_location": "cpu"}
    try:
        checkpoint = torch.load(str(weights_path), weights_only=False, **load_kwargs)
    except TypeError:
        checkpoint = torch.load(str(weights_path), **load_kwargs)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}") from exc

    # Prefer the "model" field from a training checkpoint; otherwise treat the whole object as a state_dict.
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state_dict, dict):
        raise TypeError(
            "Checkpoint format not recognised: expected a dict or checkpoint['model']."
        )

    # Strip the 'module.' prefix left by DDP.
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    info = model.load_state_dict(state_dict, strict=False)
    missing = len(info.missing_keys)
    unexpected = len(info.unexpected_keys)
    print(
        f"[ckpt] Loaded weights from {weights_path} "
        f"(missing={missing}, unexpected={unexpected})"
    )
    if missing == 0:
        print("[ckpt] All model parameters loaded successfully (no missing keys)")
    else:
        preview = info.missing_keys[:10]
        for key in preview:
            print(f"    missing: {key}")
        if missing > len(preview):
            print(f"    ... ({missing - len(preview)} more)")
    if unexpected > 0:
        preview = info.unexpected_keys[:10]
        for key in preview:
            print(f"    unexpected: {key}")
        if unexpected > len(preview):
            print(f"    ... ({unexpected - len(preview)} more)")

    # Try to overlay EMA parameters (evaluation / inference usually prefers EMA).
    ema_state = checkpoint.get("ema") or checkpoint.get("ema_state_dict")
    if ema_state is not None:
        ema_params = ema_state.get("shadow_params") or ema_state.get("average_params")
        if isinstance(ema_params, dict):
            ema_params = {k.replace("module.", ""): v for k, v in ema_params.items()}
            info_ema = model.load_state_dict(ema_params, strict=False)
            missing_ema = len(info_ema.missing_keys)
            unexpected_ema = len(info_ema.unexpected_keys)
            print(
                "[ckpt] Loaded EMA parameters "
                f"(missing={missing_ema}, unexpected={unexpected_ema})"
            )
            if missing_ema == 0:
                print("[ckpt] All EMA parameters loaded successfully (no missing keys)")
        elif isinstance(ema_params, (list, tuple)):
            # When EMA is stored as a list, align it with the key order of model.state_dict().
            model_keys = list(model.state_dict().keys())
            if len(ema_params) == len(model_keys):
                ema_dict = {
                    k: v
                    for k, v in zip(model_keys, ema_params)
                    if isinstance(v, torch.Tensor)
                }
                info_ema = model.load_state_dict(ema_dict, strict=False)
                missing_ema = len(info_ema.missing_keys)
                unexpected_ema = len(info_ema.unexpected_keys)
                print(
                    "[ckpt] Loaded EMA parameters (list format) "
                    f"(missing={missing_ema}, unexpected={unexpected_ema})"
                )
                if missing_ema == 0:
                    print(
                        "[ckpt] All EMA parameters loaded successfully (no missing keys)"
                    )
            else:
                print("[WARN] EMA list length does not match model parameters; skip EMA loading.")
        else:
            print("[WARN] EMA state format not recognised; skipping EMA loading.")

    model.eval()
    return model


__all__ = ["load_config", "instantiate_model"]
