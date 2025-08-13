from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: float, device: torch.device) -> Tensor:
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))


def karras_step(
    x: Tensor,
    sigma: Tensor,
    next_sigma: Tensor,
    denoise_fn: Callable[[Tensor, Tensor], Tensor],
    s_churn: float,
    s_tmin: float,
    s_tmax: float,
    s_noise: float,
    order: int,
    num_steps_total: int,
) -> Tensor:
    device = x.device
    dtype = x.dtype
    # Scalar gamma base per Karras et al.
    gamma_base = min(s_churn / max(num_steps_total, 1), 2 ** 0.5 - 1)
    # Elementwise gamma enabled only within [s_tmin, s_tmax]
    mask = (sigma >= s_tmin) & (sigma <= s_tmax)
    gamma = torch.zeros_like(sigma, dtype=dtype, device=device)
    if gamma_base > 0:
        gamma = torch.where(mask, torch.full_like(sigma, fill_value=gamma_base, dtype=dtype, device=device), gamma)
    sigma_hat = sigma * (1.0 + gamma)

    # Stochasticity
    if gamma_base > 0:
        eps = torch.randn_like(x) * float(s_noise)
        delta = (sigma_hat.square() - sigma.square()).clamp_min(0).sqrt().view(-1, 1, 1, 1)
        x = x + eps * delta

    denoised = denoise_fn(x, sigma_hat)
    d = (x - denoised) / sigma_hat.view(-1, 1, 1, 1)
    dt = (next_sigma - sigma_hat)

    if order == 1 or (next_sigma == 0).all().item():
        x = x + d * dt.view(-1, 1, 1, 1)
    else:
        # Heun's method
        x_euler = x + d * dt.view(-1, 1, 1, 1)
        denoised_2 = denoise_fn(x_euler, next_sigma)
        d_2 = (x_euler - denoised_2) / next_sigma.view(-1, 1, 1, 1)
        d_prime = (d + d_2) / 2.0
        x = x + d_prime * dt.view(-1, 1, 1, 1)
    return x 