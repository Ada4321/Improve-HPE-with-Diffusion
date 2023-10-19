import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction



def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class AddEmbeds(nn.Module):
    def __init__(self, embeds) -> None:
        super().__init__()
        self.embeds = embeds
        self.func = lambda x, e: x + e
    def forward(self, x):
        assert x.ndim == 3
        self.embeds = self.embeds.unsqueeze(0).expand(x.shape[0], x.shape[1], -1)
        return self.func(x, self.embeds)

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count  #(count,)
        # noise_level.unsqueeze(-1) -- (b,1,1)
        # step.unsqueeze(0) -- (1,count)
        if noise_level.ndim == 2:
            noise_level = noise_level.unsqueeze(-1)
        step = step.unsqueeze(0).expand(noise_level.shape[-1], -1) # (d, count)
        # encoding = noise_level * torch.exp(-math.log(1e4) * step)  # (B,n,1) * (1,N) => (B,n,N)
        encoding = torch.matmul(noise_level, torch.exp(-math.log(1e4) * step))  # (B,n,d) * (d,count) => (B,n,count)
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)    # (B,n,2*count)

        return encoding
    
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        count = self.dim // 2
        step = torch.arange(count, dtype=inputs.dtype,
                            device=inputs.device) / count  #(count,)
        # inputs -- (b,num_kps,num_dim)
        step = step.unsqueeze(0).expand(inputs.shape[-1], -1)  # (num_dim,count)
        encoding = torch.matmul(inputs, torch.exp(-math.log(1e4) * step))  # (b,num_kps,count)
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)    # (B,n,2N)

        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)