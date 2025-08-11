from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from .blocks import BaseDiffusionModel, UNetConfig
from .denoiser_utils import compute_conditioners, apply_noise, wrap_model_output
from .sampler_utils import build_sigmas, karras_step

@dataclass
class EDMConfig:
    unet: UNetConfig
    p_mean: float
    p_std: float
    sigma_data: float
    # Karras schedule and sampler settings
    sigma_min: float = 0.002
    sigma_max: float = 80
    rho: float = 7
    s_churn: float = 0
    s_tmin: float = 0
    s_tmax: float = float('inf')
    s_noise: float = 1
    order: int = 1
    sigma_offset_noise: float = 0.0
    quantize_output: bool = True

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
            rho=config.rho,
            s_churn=config.s_churn,
            s_tmin=config.s_tmin,
            s_tmax=config.s_tmax,
            s_noise=config.s_noise,
            order=config.order,
            sigma_offset_noise=config.sigma_offset_noise,
            quantize_output=config.quantize_output,
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
        rho: float = 7,
        s_churn: float = 0,
        s_tmin: float = 0,
        s_tmax: float = float('inf'),
        s_noise: float = 1,
        order: int = 1,
        sigma_offset_noise: float = 0.0,
        quantize_output: bool = True,
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
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.order = order
        self.sigma_offset_noise = sigma_offset_noise
        self.quantize_output = quantize_output

    def _denoise(self, x: torch.Tensor, sigma: torch.Tensor, prev_frames: torch.Tensor, prev_actions: torch.Tensor):
        # Build conditioners with offset noise
        sigma = sigma.to(torch.float32).flatten()
        cs = compute_conditioners(sigma, self.sigma_data, self.sigma_offset_noise)
        # Prepare model input
        noise_imgs = torch.concat([(cs.c_in * x)[:, None, :, :, :], prev_frames], dim=1).flatten(1,2)
        F_x = self.model(noise_imgs, cs.c_noise.flatten(), prev_actions)
        # Wrap output, optional quantization enabled only in sampling paths
        D_x = wrap_model_output(x, F_x.to(torch.float32), cs, quantize_output=self.quantize_output)
        return D_x

    def forward(self, imgs: torch.Tensor, prev_frames: torch.Tensor, prev_actions: torch.Tensor):
        assert prev_frames.shape[1] == prev_actions.shape[1] == self.context_length
        # Sample sigmas log-normal per EDM
        rnd_normal = torch.randn([imgs.shape[0]], device=self.device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        # Apply noise with optional offset noise
        noisy = apply_noise(imgs, sigma, self.sigma_offset_noise)
        # Compute conditioners and target per Karras
        cs = compute_conditioners(sigma, self.sigma_data, self.sigma_offset_noise)
        # Get raw model output in model space (no quantization, no wrapping)
        noise_imgs = torch.concat([(cs.c_in * noisy)[:, None, :, :, :], prev_frames], dim=1).flatten(1,2)
        model_out = self.model(noise_imgs, cs.c_noise.flatten(), prev_actions).to(torch.float32)
        # Target in model space
        target = (imgs - cs.c_skip * noisy) / cs.c_out
        loss = F.mse_loss(model_out, target)
        return loss
        
    @torch.no_grad()
    def sample(
        self, steps: int, size: Tuple[int],
        prev_frames: torch.Tensor,
        prev_actions: torch.Tensor
    ) -> torch.Tensor:
        # Initial image is pure noise
        x = torch.randn(1, *size, device=self.device)
        sigmas = build_sigmas(steps, self.sigma_min, self.sigma_max, self.rho, self.model.parameters().__next__().device)
        # Precompute scalar gamma as in Karras; when steps>=2, use s_churn/(steps-1), capped
        denom = max(1, steps - 1)
        gamma_scalar = min(self.s_churn / denom, 2 ** 0.5 - 1) if self.s_churn > 0 else 0.0

        def denoise_fn(x_in: torch.Tensor, sigma_in: torch.Tensor) -> torch.Tensor:
            return self._denoise(x_in, sigma_in, prev_frames, prev_actions).to(torch.float)

        for i in range(len(sigmas) - 1):
            sigma = sigmas[i].repeat(x.shape[0])
            next_sigma = sigmas[i + 1].repeat(x.shape[0])
            x = karras_step(
                x,
                sigma,
                next_sigma,
                denoise_fn,
                gamma_scalar,
                self.s_tmin,
                self.s_tmax,
                self.s_noise,
                self.order,
            )
        return x
        
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