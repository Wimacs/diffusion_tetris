from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class Conditioners:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor


def compute_conditioners(sigma: Tensor, sigma_data: float, sigma_offset_noise: float) -> Conditioners:
    # Adjust sigma with offset noise per Karras/EDM practice
    sigma = (sigma**2 + sigma_offset_noise**2).sqrt()
    c_in = 1 / (sigma**2 + sigma_data**2).sqrt()
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * c_skip.sqrt()
    c_noise = sigma.log() / 4
    return Conditioners(*(add_dims(c, n) for c, n in zip((c_in, c_out, c_skip, c_noise), (4, 4, 4, 1))))


def apply_noise(x: Tensor, sigma: Tensor, sigma_offset_noise: float) -> Tensor:
    # x: (B, C, H, W), sigma: (B,)
    b, c, _, _ = x.shape
    device = x.device
    if sigma_offset_noise != 0:
        offset_noise = sigma_offset_noise * torch.randn(b, c, 1, 1, device=device)
    else:
        offset_noise = 0
    return x + offset_noise + torch.randn_like(x) * add_dims(sigma, x.ndim)


def wrap_model_output(noisy_x: Tensor, model_output: Tensor, cs: Conditioners, quantize_output: bool = True) -> Tensor:
    # Denoised estimate D(x) = c_skip * noisy + c_out * model_output
    d = cs.c_skip * noisy_x + cs.c_out * model_output
    if quantize_output:
        # Quantize to {0, ..., 255}, then back to [-1, 1]
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
    return d 