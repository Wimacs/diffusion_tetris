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

# ===== New blocks to match clipboard UNet style =====

class AdaGroupNorm(nn.Module):
    def __init__(self, in_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = max(1, in_channels // GROUP_SIZE)
        self.linear = nn.Linear(cond_channels, in_channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        assert x.size(1) == self.in_channels
        x = F.group_norm(x, self.num_groups, eps=GN_EPS)
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift

class SelfAttention2d(nn.Module):
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

class FourierFeatures(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer("weight", torch.randn(1, cond_channels // 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 1
        f = 2 * math.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)

class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class SmallResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.f = nn.Sequential(GroupNorm(in_channels), nn.SiLU(inplace=True), Conv3x3(in_channels, out_channels))
        self.skip_projection = nn.Identity() if in_channels == out_channels else Conv1x1(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip_projection(x) + self.f(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_channels: int, attn: bool) -> None:
        super().__init__()
        should_proj = in_channels != out_channels
        self.proj = Conv1x1(in_channels, out_channels) if should_proj else nn.Identity()
        self.norm1 = AdaGroupNorm(in_channels, cond_channels)
        self.conv1 = Conv3x3(in_channels, out_channels)
        self.norm2 = AdaGroupNorm(out_channels, cond_channels)
        self.conv2 = Conv3x3(out_channels, out_channels)
        self.attn = SelfAttention2d(out_channels) if attn else nn.Identity()
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        r = self.proj(x)
        x = self.conv1(F.silu(self.norm1(x, cond)))
        x = self.conv2(F.silu(self.norm2(x, cond)))
        x = x + r
        x = self.attn(x)
        return x

class ResBlocks(nn.Module):
    def __init__(
        self,
        list_in_channels: List[int],
        list_out_channels: List[int],
        cond_channels: int,
        attn: bool,
    ) -> None:
        super().__init__()
        assert len(list_in_channels) == len(list_out_channels)
        self.in_channels = list_in_channels[0]
        self.resblocks = nn.ModuleList(
            [
                ResBlock(in_ch, out_ch, cond_channels, attn)
                for (in_ch, out_ch) in zip(list_in_channels, list_out_channels)
            ]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, to_cat: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        outputs: List[torch.Tensor] = []
        for i, resblock in enumerate(self.resblocks):
            x = x if to_cat is None else torch.cat((x, to_cat[i]), dim=1)
            x = resblock(x, cond)
            outputs.append(x)
        return x, outputs

# ===== Legacy attention (kept to avoid breaking imports) =====

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

# ===== Old residual stack kept (unused by new UNet) =====

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
    # Support both legacy fields and new clipboard-style fields
    # New style
    cond_channels: int
    channels: List[int]
    depths: Optional[List[int]] = None
    attn_depths: Optional[List[bool]] = None
    # Legacy style (will be mapped to new)
    steps: Optional[List[int]] = None
    attn_step_indexes: Optional[List[bool]] = None

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
        # Map legacy fields to new ones if needed
        depths = config.depths if config.depths is not None else (config.steps if config.steps is not None else [2,2,2,2])
        attn_depths = config.attn_depths if config.attn_depths is not None else (config.attn_step_indexes if config.attn_step_indexes is not None else [False]*len(depths))
        return cls(
            in_channels=in_channels,
            out_channels=out_channels,
            T=T,
            actions_count=actions_count,
            seq_length=seq_length,
            depths=depths,
            channels=config.channels,
            cond_channels=config.cond_channels,
            attn_depths=attn_depths
        )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        T: Optional[int],
        actions_count: int,
        seq_length: int,
        depths: List[int] = (2, 2, 2, 2),
        channels: List[int] = (64, 64, 64, 64),
        cond_channels: int = 256,
        attn_depths: List[bool] = (False, False, False, False)
    ):
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.seq_length = seq_length
        self.T = T

        # Condition embedding: FourierFeatures for time/noise + action embedding, then MLP
        self.time_fourier = FourierFeatures(cond_channels)
        self.actions_embedding = nn.Sequential(
            nn.Embedding(actions_count, cond_channels // seq_length),
            nn.Flatten()
        )
        self.cond_embedding = nn.Sequential(
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(inplace=True),
            nn.Linear(cond_channels, cond_channels),
        )

        # Input/Output projections to decouple data channels from feature channels
        self.input_proj = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        self.output_proj = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=channels[0], out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        
        # Build encoder/decoder stacks in clipboard style
        self._num_down = len(channels) - 1
        d_blocks: List[ResBlocks] = []
        u_blocks: List[ResBlocks] = []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            d_blocks.append(
                ResBlocks(
                    list_in_channels=[c1] + [c2] * (n - 1),
                    list_out_channels=[c2] * n,
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                )
            )
            u_blocks.append(
                ResBlocks(
                    list_in_channels=[2 * c2] * n + [c1 + c2],
                    list_out_channels=[c2] * n + [c1],
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                )
            )
        self.d_blocks = nn.ModuleList(d_blocks)
        self.u_blocks = nn.ModuleList(reversed(u_blocks))

        self.mid_blocks = ResBlocks(
            list_in_channels=[channels[-1]] * 2,
            list_out_channels=[channels[-1]] * 2,
            cond_channels=cond_channels,
            attn=True,
        )

        downsamples = [nn.Identity()] + [Downsample(c) for c in channels[:-1]]
        upsamples = [nn.Identity()] + [Upsample(c) for c in reversed(channels[:-1])]
        self.downsamples = nn.ModuleList(downsamples)
        self.upsamples = nn.ModuleList(upsamples)

    def _build_condition(self, t: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        # t can be EDM continuous noise encoding (already log-scaled) or DDPM integer step
        if t.ndim == 2 and t.size(1) == 1:
            t = t.squeeze(1)
        t = t.to(dtype=torch.float32)
        if self.T is not None:
            # normalize t for DDPM if it looks like integer steps
            # avoid division by zero if T==0 (not expected)
            t = t / float(max(self.T, 1))
        time_emb = self.time_fourier(t)
        actions_emb = self.actions_embedding(prev_actions)
        cond = time_emb + actions_emb
        cond = self.cond_embedding(cond)
        return cond

    def forward(self, x: torch.Tensor, t: torch.Tensor, prev_actions: torch.Tensor):
        # Sanitize action indices for embedding
        if prev_actions.dtype != torch.long:
            prev_actions = prev_actions.to(torch.long)
        num_actions = self.actions_embedding[0].num_embeddings
        prev_actions = prev_actions.clamp(min=0, max=num_actions - 1)

        # Build condition vector
        cond = self._build_condition(t, prev_actions)

        # Project input to feature channels
        x = self.input_proj(x)

        # Auto padding to be divisible by 2^n
        *_, h, w = x.size()
        n = self._num_down
        padding_h = math.ceil(h / 2 ** n) * 2 ** n - h
        padding_w = math.ceil(w / 2 ** n) * 2 ** n - w
        if padding_h > 0 or padding_w > 0:
            x = F.pad(x, (0, padding_w, 0, padding_h))

        # Encoder
        d_outputs: List[List[torch.Tensor]] = []
        for block, down in zip(self.d_blocks, self.downsamples):
            x_down = down(x)
            x, block_outputs = block(x_down, cond)
            d_outputs.append([x_down, *block_outputs])

        # Mid
        x, _ = self.mid_blocks(x, cond)

        # Decoder (with fine-grained skip connections)
        for block, up, skip in zip(self.u_blocks, self.upsamples, reversed(d_outputs)):
            x_up = up(x)
            x, _ = block(x_up, cond, skip[::-1])

        # Crop back
        x = x[..., :h, :w]

        # Project to output channels
        x = self.output_proj(x)
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
