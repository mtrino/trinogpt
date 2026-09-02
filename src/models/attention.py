import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Model dimension should be divisible by number of attention heads"
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_SCALE_INIT = 1.0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.use_flash = config.use_flashattn
        self.register_buffer("mask", torch.tril(torch.ones(config.context_length, config.context_length))
                                            .view(1, 1, config.context_length, config.context_length))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        if self.use_flash:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            attn = (q @ k.transpose(-1, -2)) / math.sqrt(k.size(-1))
            attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

