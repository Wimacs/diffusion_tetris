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
    gamma: float,
    s_tmin: float,
    s_tmax: float,
    s_noise: float,
    order: int,
) -> Tensor:
    device = x.device
    # Make scalar tensors on correct device/dtype
    gamma_t = sigma.new_tensor(gamma)
    s_tmin_t = sigma.new_tensor(s_tmin)
    s_tmax_t = sigma.new_tensor(s_tmax)
    s_noise_t = sigma.new_tensor(s_noise)

    # Apply churn only within [s_tmin, s_tmax]
    mask = (sigma >= s_tmin_t) & (sigma <= s_tmax_t)
    gamma_vec = torch.where(mask, gamma_t, sigma.new_zeros(()))

    sigma_hat = sigma * (gamma_vec + 1)

    if (gamma_vec > 0).any():
        eps = torch.randn_like(x) * s_noise_t
        x = x + eps * (sigma_hat.square() - sigma.square()).clamp_min(0).sqrt().view(-1, 1, 1, 1)

    denoised = denoise_fn(x, sigma_hat)
    d = (x - denoised) / sigma_hat.view(-1, 1, 1, 1)
    dt = (next_sigma - sigma_hat).view(-1, 1, 1, 1)

    if order == 1 or (next_sigma == 0).all():
        x = x + d * dt
    else:
        # Heun
        x_euler = x + d * dt
        denoised_2 = denoise_fn(x_euler, next_sigma)
        d_2 = (x_euler - denoised_2) / next_sigma.view(-1, 1, 1, 1)
        d_prime = (d + d_2) / 2
        x = x + d_prime * dt
    return x 