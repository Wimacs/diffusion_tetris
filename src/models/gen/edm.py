from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from .blocks import BaseDiffusionModel, UNetConfig

@dataclass
class EDMConfig:
    unet: UNetConfig
    p_mean: float
    p_std: float
    sigma_data: float
    sigma_min: float = 0.002
    sigma_max: float = 80
    rho: float = 7

# Took parts of the code from the official implementation of the paper:
# https://github.com/NVlabs/edm

class EDM(BaseDiffusionModel):
    @classmethod
    def from_config(
        cls,
        config: EDMConfig,
        context_length: int,
        device: str,
        model: nn.Module
    ):
        return cls(
            p_mean=config.p_mean,
            p_std=config.p_std,
            sigma_data=config.sigma_data,
            model=model,
            device=device,
            context_length = context_length,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            rho=config.rho
        )

    def __init__(
        self,
        p_mean: float,
        p_std: float,
        sigma_data: float,
        model: nn.Module,
        context_length: int,
        device: str,
        sigma_min = 0.002,           
        sigma_max = 80,
        rho: float = 7
    ):
        super().__init__()
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data
        self.model = model.to(device)
        self.device = device
        self.context_length = context_length
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def _denoise(self, x: torch.Tensor, sigma: torch.Tensor, prev_frames: torch.Tensor, prev_actions: torch.Tensor):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        # Concat noise imgs with previous frames
        noise_imgs = torch.concat([(c_in * x)[:, None, :, :, :], prev_frames], dim=1).flatten(1,2)

        F_x = self.model(noise_imgs, c_noise.flatten(), prev_actions)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def forward(self, imgs: torch.Tensor, prev_frames: torch.Tensor, prev_actions: torch.Tensor):
        assert prev_frames.shape[1] == prev_actions.shape[1] == self.context_length

        rnd_normal = torch.randn([imgs.shape[0], 1, 1, 1], device=self.device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = imgs
        n = torch.randn_like(y) * sigma

        D_yn = self._denoise(y + n, sigma, prev_frames, prev_actions)
        
        return (weight * ((D_yn - y) ** 2)).mean()
        
    @torch.no_grad()
    def sample(
        self, steps: int, size: Tuple[int],
        prev_frames: torch.Tensor,
        prev_actions: torch.Tensor
    ) -> torch.Tensor:
        x_t = torch.randn(1, *size, device=self.device)

        # Adjust noise levels based on what's supported by the network.
        sigma_min = self.sigma_min
        sigma_max = self.sigma_max
        rho = self.rho

        # Time step discretization.
        step_indices = torch.arange(steps, dtype=torch.float, device=self.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        # Main sampling loop.
        x_next = x_t.to(torch.float) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_hat = x_next
            t_hat = t_cur
            
            # Euler step.
            denoised = self._denoise(x_hat, t_hat, prev_frames, prev_actions).to(torch.float)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < steps - 1:
                denoised = self._denoise(x_next, t_next, prev_frames, prev_actions).to(torch.float)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next
        
if __name__ == "__main__":
    size = (64, 64)
    input_channels = 3
    context_length = 4
    actions_count = 5
    batch_size = 3

    from blocks import UNet

    unet = UNet((input_channels) * (context_length + 1), 3, None, actions_count, context_length)
    ddpm = EDM(
        p_mean=-1.2,
        p_std=1.2,
        sigma_data=0.5,
        model=unet,
        context_length=context_length,
        device="cpu"
    )

    img = torch.randn((batch_size, input_channels, *size))
    prev_frames = torch.randn((batch_size, context_length, input_channels, *size))
    # frames = torch.concat([img[:, None, :, :, :], prev_frames], dim=1).flatten(1,2)

    prev_actions = torch.randint(low=0, high=actions_count, size=(batch_size, context_length))
    ddpm.forward(img, prev_frames, prev_actions)