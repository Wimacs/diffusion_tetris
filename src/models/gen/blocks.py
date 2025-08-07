from typing import List, Optional, Tuple
from functools import partial
import abc
from abc import ABC
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class BaseDiffusionModel(nn.Module, ABC):
    @torch.no_grad()
    @abc.abstractmethod
    def sample(self, steps: int, size: Tuple[int],
        prev_frames: torch.Tensor,
        prev_actions: torch.Tensor
    ) -> torch.Tensor:
        pass
    
class PositionalEmbedding(nn.Module):
    def __init__(self, T: int, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        position = torch.arange(T+1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, output_dim, 2) * (-math.log(10000.0) / output_dim))
        pe = torch.zeros(T+1, output_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        return self.pe[x].reshape(x.shape[0], self.output_dim)
    
class FloatPositionalEmbedding(nn.Module):
    def __init__(self, output_dim: int, max_positions=10000) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.max_positions = max_positions

    def forward(self, x: torch.Tensor):
        freqs = torch.arange(start=0, end=self.output_dim//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.output_dim // 2 - 1)
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
    
Conv1x1 = partial(nn.Conv2d, kernel_size=1, stride=1, padding=0)
Conv3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)

# GroupNorm and conditional GroupNorm

GROUP_SIZE = 32
GN_EPS = 1e-5

class GroupNorm(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        num_groups = max(1, in_channels // GROUP_SIZE)
        self.norm = nn.GroupNorm(num_groups, in_channels, eps=GN_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

class MultiheadAttention(nn.Module):
    def __init__(self, in_channels: int, head_dim: int = 8) -> None:
        super().__init__()
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0
        self.norm = GroupNorm(in_channels)
        self.qkv_proj = Conv1x1(in_channels, in_channels * 3)
        self.out_proj = Conv1x1(in_channels, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.qkv_proj(x)
        qkv = qkv.view(n, self.n_head * 3, c // self.n_head, h * w).transpose(2, 3).contiguous()
        q, k, v = [x for x in qkv.chunk(3, dim=1)]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(2, 3).reshape(n, c, h, w)
        return x + self.out_proj(y)

class NormBlock(nn.Module):
    def __init__(self, in_channels, cond_channels):
        super().__init__()
        self.norm = nn.GroupNorm(max(in_channels // GROUP_SIZE, 1), in_channels)
        self.ln = nn.Linear(cond_channels, in_channels)

    def forward(self, x, cond):
        return self.norm(x) + self.ln(cond)[:, :, None, None]

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels, has_attn=False):
        super().__init__()
        self.proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.norm_1 = NormBlock(out_channels, cond_channels)
        self.conv_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        )
        self.norm_2 = NormBlock(out_channels, cond_channels)
        self.conv_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        )
        
        self.attn = nn.Identity() if not has_attn else MultiheadAttention(out_channels)

    def forward(self, x, cond):
        h = self.proj(x)
        x = self.conv_1(self.norm_1(h, cond))
        x = self.conv_2(self.norm_2(x, cond))
        return self.attn(h + x)
    
class DownBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.pool = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.pool(x)
    
class UpBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class ResnetsBlock(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels_list: List[int], cond_channels: int, has_attn=False):
        super().__init__()
        assert len(in_channels_list) == len(out_channels_list)
        self.models = nn.ModuleList([
            ResnetBlock(in_ch, out_ch, cond_channels, has_attn) for in_ch, out_ch in zip(in_channels_list, out_channels_list)
        ])
    
    def forward(self, x, cond):
        for module in self.models:
            x = module(x, cond)
        return x
    
@dataclass
class UNetConfig:
    steps: List[int]
    channels: List[int]
    cond_channels: int
    attn_step_indexes: List[bool]

class UNet(nn.Module):
    @classmethod
    def from_config(
        cls,
        config: UNetConfig,
        in_channels: int,
        out_channels: int,
        actions_count: int,
        seq_length: int,
        T: Optional[int] = None
    ):
        return cls(
            in_channels=in_channels,
            out_channels=out_channels,
            T=T,
            actions_count=actions_count,
            seq_length=seq_length,
            steps=config.steps,
            channels=config.channels,
            cond_channels=config.cond_channels,
            attn_step_indexes=config.attn_step_indexes
        )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        T: Optional[int],
        actions_count: int,
        seq_length: int,
        steps=(2, 2, 2, 2),
        channels = (64, 64, 64, 64),
        cond_channels = 256,
        attn_step_indexes = [False, False, False, False]
    ):
        super().__init__()
        assert len(steps) == len(channels) == len(attn_step_indexes)
        self.time_embedding = PositionalEmbedding(T=T, output_dim=cond_channels) if T is not None else FloatPositionalEmbedding(output_dim=cond_channels)
        self.actions_embedding = nn.Sequential(
            nn.Embedding(actions_count, cond_channels // seq_length),
            nn.Flatten()
        )
        self.cond_embedding = nn.Sequential(
            nn.Linear(cond_channels, cond_channels),
            nn.ReLU(),
            nn.Linear(cond_channels, cond_channels),
        )

        self.first_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        down_res_blocks = []
        self.downsample_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        up_res_blocks = []
        for (index, step) in enumerate(steps):
            in_ch = step * [channels[index]]
            out_ch = in_ch.copy()
            out_ch[-1] = channels[index + 1] if index < len(steps) - 1 else channels[index]
            down_res_blocks.append(ResnetsBlock(
                in_channels_list=in_ch,
                out_channels_list=out_ch,
                cond_channels=cond_channels,
                has_attn=index == attn_step_indexes[index]
            ))
            self.downsample_blocks.append(DownBlock(in_ch[0]))
            self.upsample_blocks.append(UpBlock(out_ch[-1]))
            in_ch = step * [channels[index]]
            out_ch = in_ch.copy()
            in_ch[0] = 2 * (channels[index + 1] if index < len(steps) - 1 else channels[index])
            up_res_blocks.append(ResnetsBlock(
                in_channels_list=in_ch,
                out_channels_list=out_ch,
                cond_channels=cond_channels,
                has_attn=index == attn_step_indexes[index]
            ))
        self.downres_blocks = nn.ModuleList(down_res_blocks)
        self.upres_blocks = nn.ModuleList(reversed(up_res_blocks))
        self.backbone = ResnetsBlock(
            [channels[-1]] * 2,
            [channels[-1]] * 2,
            cond_channels=cond_channels,
            has_attn=True
        )
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[0], out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, prev_actions: torch.Tensor):
        assert x.shape[0] == prev_actions.shape[0]
        time_emb = self.time_embedding(t)
        actions_emb = self.actions_embedding(prev_actions)
        cond = self.cond_embedding(time_emb + actions_emb)

        x = self.first_conv(x)
        hx = []
        for index, downres_block in enumerate(self.downres_blocks):
            x = downres_block(x, cond)
            hx.append(x)
            x = self.downsample_blocks[index](x)                
        x = self.backbone(x, cond)

        for index, up_block in enumerate(self.upres_blocks):
            x = self.upsample_blocks[len(self.upres_blocks) - index - 1](x)
            x = up_block(torch.cat([x, hx[len(self.upres_blocks) - index - 1]], 1), cond)
        x = self.out(x)

        return x
    
if __name__ == "__main__":
    DownBlock(4).forward(torch.rand(4,4,64,64)).size(-1) == 32
    DownBlock(4).forward(torch.rand(4,4,32,32)).size(-1) == 16

    size = (64, 64)
    input_channels = 3
    context_length = 4
    actions_count = 5
    T = 1000
    batch_size = 2
    unet = UNet(
        (input_channels) * (context_length + 1),
        3,
        None,
        actions_count,
        context_length
    )
    img = torch.randn((batch_size, input_channels, *size))
    prev_frames = torch.randn((batch_size, context_length, input_channels, *size))
    frames = torch.concat([img[:, None, :, :, :], prev_frames], dim=1).flatten(1,2)

    prev_actions = torch.randint(low=0, high=actions_count, size=(batch_size, context_length))
    t = torch.randn((batch_size,))
    unet.forward(frames, t.unsqueeze(1), prev_actions)
