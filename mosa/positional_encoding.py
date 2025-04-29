from typing import Tuple

import torch
from torch import nn

class RotaryPosEncoding(nn.Module):
    # RoPE based on: https://www.kaggle.com/code/aeryss/rotary-postional-encoding-rope-pytorch
    def __init__(self, d_model: int, base=10000, seq_dim: int = 1):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError("RoPE can only be used with an even number of dimensions")

        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = 0
        self.cos_cached = None
        self.sin_cached = None
        self.seq_dim = seq_dim

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat(
            (-x2, x1), dim=x1.ndim - 1
        )  
    
    def cache_enc(self, seq_len):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        self.emb = torch.cat((freqs, freqs), dim=-1)

    def apply_rot(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        sin = sin.narrow(self.seq_dim, 0, x.shape[self.seq_dim])
        cos = cos.narrow(self.seq_dim, 0, x.shape[self.seq_dim])
        return (x * cos) + (self.rotate_half(x) * sin)

    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply_rot(q, sin, cos), self.apply_rot(k, sin, cos)

    def get(self, x: torch.Tensor, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if max_len > self.max_seq_len:
            self.cache_enc(max_len)
            tgt_shape = [1] * x.ndim
            tgt_shape[self.seq_dim] = self.max_seq_len
            tgt_shape[-1] = x.shape[-1]

            self.cos_cached = self.emb.cos().view(*tgt_shape)
            self.sin_cached = self.emb.sin().view(*tgt_shape)
        if self.emb.device != x.device:
            self.emb = self.emb.to(x.device)

        return self.sin_cached, self.cos_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = k.shape[-2]
        sin, cos = self.get(k, T)
        return self.apply_rotary_pos_emb(q, k, sin, cos)

class MoSARotaryPosEncoding(RotaryPosEncoding):
    def apply_rot(self, x: torch.Tensor, Ind: torch.Tensor,
                  sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        # x.shape = [b, E, k, h]
        # x_I.shape = [b, E, k] has index from 0 to max_seq_len
        # sin.shape = [1,1,k,hid_dim]
        Ind = Ind.unsqueeze(-1).expand(*x.shape) # add dimension for latent dim

        sin_new_shape = (x.shape[0], x.shape[1], self.max_seq_len, sin.shape[-1])
        sin_gathered = torch.gather(sin.expand(sin_new_shape), dim=self.seq_dim, index=Ind)
        cos_gathered = torch.gather(cos.expand(sin_new_shape), dim=self.seq_dim, index=Ind)

        y = (x * cos_gathered) + (self.rotate_half(x) * sin_gathered)
        return y   
    def apply_rotary_pos_emb(self, q: torch.Tensor, q_I: torch.Tensor, k: torch.Tensor, k_I: torch.Tensor,
                             sin: torch.Tensor, cos: torch.Tensor
                             )-> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply_rot(q, q_I, sin, cos), self.apply_rot(k, k_I, sin, cos)
    
    def forward(self, q: torch.Tensor, q_I: torch.Tensor, k: torch.Tensor, k_I: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        sin, cos = self.get(k, q_I.max()+1)
        return self.apply_rotary_pos_emb(q, q_I, k, k_I, sin, cos)
