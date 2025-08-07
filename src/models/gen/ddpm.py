from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from .blocks import BaseDiffusionModel, UNetConfig

@dataclass
class DDPMConfig:
    T: int
    unet: UNetConfig

class DDPM(BaseDiffusionModel):
    @classmethod
    def from_config(
        cls,
        config: DDPMConfig,
        context_length: int,
        device: str,
        model: nn.Module
    ):
        return cls(
            T=config.T,
            context_length=context_length,
            device=device,
            model=model
        )

    def __init__(
        self,
        T: int,
        context_length: int,
        device: str,
        model: nn.Module
    ):
        super().__init__()
        self.T = T
        self.eps_model = model.to(device)
        self.device = device
        self.context_length = context_length
        beta_schedule = torch.linspace(1e-4, 0.02, T + 1, device=self.device)
        alpha_t_schedule = 1 - beta_schedule
        bar_alpha_t_schedule = torch.cumprod(alpha_t_schedule.detach().cpu(), 0).to(self.device)
        sqrt_bar_alpha_t_schedule = torch.sqrt(bar_alpha_t_schedule)
        sqrt_minus_bar_alpha_t_schedule = torch.sqrt(1 - bar_alpha_t_schedule)
        self.register_buffer("beta_schedule", beta_schedule)
        self.register_buffer("alpha_t_schedule", alpha_t_schedule)
        self.register_buffer("bar_alpha_t_schedule", bar_alpha_t_schedule)
        self.register_buffer("sqrt_bar_alpha_t_schedule", sqrt_bar_alpha_t_schedule)
        self.register_buffer("sqrt_minus_bar_alpha_t_schedule", sqrt_minus_bar_alpha_t_schedule)
        self.criterion = nn.MSELoss()        

    def forward(self, imgs: torch.Tensor, prev_frames: torch.Tensor, prev_actions: torch.Tensor):
        assert prev_frames.shape[1] == prev_actions.shape[1] == self.context_length
        t = torch.randint(low=1, high=self.T+1, size=(imgs.shape[0],), device=self.device)
        noise = torch.randn_like(imgs, device=self.device)
        batch_size, channels, width, height = imgs.shape
        # Add noise to the image
        noise_imgs = self.sqrt_bar_alpha_t_schedule[t].view((batch_size, 1, 1 ,1)) * imgs \
            + self.sqrt_minus_bar_alpha_t_schedule[t].view((batch_size, 1, 1, 1)) * noise
        # Concat noise imgs with previous frames
        noise_imgs = torch.concat([noise_imgs[:, None, :, :, :], prev_frames], dim=1).flatten(1,2)

        pred_noise = self.eps_model(noise_imgs, t.unsqueeze(1), prev_actions)

        return self.criterion(pred_noise, noise)
    
    @torch.no_grad()
    def sample(
        self,
        size: Tuple[int],
        prev_frames: torch.Tensor,
        prev_actions: torch.Tensor
    ):
        x_t = torch.randn(1, *size, device=self.device)
        for t in range(self.T, 0, -1):
            z = torch.randn_like(x_t, device=self.device) if t > 0 else 0
            t_tensor = torch.tensor([t], device=self.device).repeat(x_t.shape[0], 1)
            big_x_t = torch.concat([x_t[:, None, :, :, :], prev_frames], dim=1).flatten(1,2)
            pred_noise = self.eps_model(big_x_t, t_tensor, prev_actions)
            
            x_t = 1 / torch.sqrt(self.alpha_t_schedule[t]) * \
                (x_t - pred_noise * (1 - self.alpha_t_schedule[t]) / self.sqrt_minus_bar_alpha_t_schedule[t]) + \
                torch.sqrt(self.beta_schedule[t]) * z
        return x_t
        
    # DDIM implementation
    @torch.no_grad()
    def sample(
        self, steps: int, size: Tuple[int],
        prev_frames: torch.Tensor,
        prev_actions: torch.Tensor
    ) -> torch.Tensor:
        x_t = torch.randn(1, *size, device=self.device)
        step_size = self.T // steps
        range_t = range(self.T, -1, -step_size)
        next_range_t = range_t[1:]
        range_t = range_t[:-1]
        for i, j in zip(range_t, next_range_t):
            t_tensor = torch.tensor([i], device=self.device).repeat(x_t.shape[0], 1)

            big_x_t = torch.concat([x_t[:, None, :, :, :], prev_frames], dim=1).flatten(1,2)
            pred_noise = self.eps_model(big_x_t, t_tensor, prev_actions)
            alpha = self.bar_alpha_t_schedule[i]
            next_alpha = self.bar_alpha_t_schedule[j]
            x0 = (x_t - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
            new_xt = torch.sqrt(1 - next_alpha) * pred_noise

            x_t = torch.sqrt(next_alpha) * x0 + new_xt
        return x_t
        
if __name__ == "__main__":
    size = (64, 64)
    input_channels = 3
    context_length = 4
    actions_count = 5
    T = 1000
    batch_size = 3

    from blocks import UNet

    unet = UNet((input_channels) * (context_length + 1), 3, T, actions_count, context_length)
    ddpm = DDPM(
        T=T,
        eps_model=unet,
        context_length=context_length,
        device="cpu"
    )

    img = torch.randn((batch_size, input_channels, *size))
    prev_frames = torch.randn((batch_size, context_length, input_channels, *size))
    # frames = torch.concat([img[:, None, :, :, :], prev_frames], dim=1).flatten(1,2)

    prev_actions = torch.randint(low=0, high=actions_count, size=(batch_size, context_length))
    t = torch.randint(1, T + 1, (batch_size,))
    ddpm.forward(img, prev_frames, prev_actions)