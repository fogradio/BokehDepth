import torch
import torch.nn as nn

from .utils import FNS


class SILog(nn.Module):
    def __init__(
        self,
        weight: float,
        input_fn: str = "linear",
        output_fn: str = "sqrt",
        integrated: float = 0.15,
        dims: list[int] = [-3, -2, -1],
        eps: float = 1e-5,
    ):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.weight: float = weight

        self.dims = dims
        self.input_fn = FNS[input_fn]
        self.output_fn = FNS[output_fn]
        self.eps: float = eps
        self.integrated = integrated

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        si: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        input = input.float()
        target = target.float()

        finite_mask = torch.isfinite(input) & torch.isfinite(target)
        input = torch.clamp(input, min=self.eps)
        target = torch.clamp(target, min=self.eps)

        mask = mask.bool() & finite_mask
        if not torch.any(mask):
            return torch.zeros(
                input.shape[0] if input.ndim > 1 else 1,
                device=input.device,
                dtype=input.dtype,
            )

        error = self.input_fn(input) - self.input_fn(target)
        error = torch.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)

        dim_count = error.ndim
        reduce_dims = tuple(
            sorted(
                {
                    d if d >= 0 else dim_count + d
                    for d in self.dims
                    if -(dim_count) <= d < dim_count
                }
            )
        )
        mask_float = mask.float()
        valid_per_axis = mask_float.sum(dim=reduce_dims)
        clamped_valid = valid_per_axis.clamp(min=1.0)

        mean_error = (error * mask_float).sum(dim=reduce_dims) / clamped_valid
        sq_error_mean = (error.square() * mask_float).sum(dim=reduce_dims) / clamped_valid

        valid_entries = (valid_per_axis > 0)
        mean_error = torch.where(valid_entries, mean_error, torch.zeros_like(mean_error))
        sq_error_mean = torch.where(
            valid_entries, sq_error_mean, torch.zeros_like(sq_error_mean)
        )

        var_error = torch.clamp(sq_error_mean - mean_error.square(), min=0.0)

        if mean_error.ndim > 1:
            mean_error = mean_error.reshape(mean_error.shape[0], -1).mean(dim=-1)
        if var_error.ndim > 1:
            var_error = var_error.reshape(var_error.shape[0], -1).mean(dim=-1)

        if self.integrated > 0.0:
            scale_error = mean_error**2
            var_error = var_error + self.integrated * scale_error * (1 - si.int())

        var_error = torch.where(
            var_error.isfinite(), var_error, torch.zeros_like(var_error)
        )
        var_error = torch.clamp(var_error, min=0.0)

        out_loss = self.output_fn(var_error)
        return torch.nan_to_num(out_loss, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            dims=config["dims"],
            output_fn=config["output_fn"],
            input_fn=config["input_fn"],
            integrated=config.get("integrated", 0.15),
        )
        return obj
