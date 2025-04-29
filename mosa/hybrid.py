from typing import Tuple

import torch 
from torch import nn
from torch.nn import functional as F

from .positional_encoding import RotaryPosEncoding
from .MoSA import PureMoSA

class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, h_prim: int, n_heads: int):
        super().__init__()

        self.h, self.h_prim, self.n_heads = h, h_prim, n_heads

        self.W_QKV = nn.Linear(h, 3*h_prim*n_heads, bias=False)
        self.W_O = nn.Linear(h_prim*n_heads, h, bias=False)

    def inner_attend(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        raise NotImplementedError()

    def forward(self, X: torch.Tensor):
        b, T, _ = X.shape

        Q, K, V = self.W_QKV(X).reshape(b, T, self.n_heads, 3*self.h_prim).transpose(1,2).chunk(3, dim=-1)
        AV = self.inner_attend(Q, K, V) # [b, E, T, h_prim]
        return self.W_O(AV.transpose(1,2).reshape(b, T, -1))


class Dense(MultiHeadAttention):
    def __init__(self, h: int, h_prim: int, n_heads: int, 
                 rotate_fraction: float = 0.5, rope_base: float=10000
                 ):
        super().__init__(h, h_prim, n_heads)

        # rope
        self.n_rotate = int(rotate_fraction * self.h_prim)
        self.n_rotate -= self.n_rotate % 2
        if self.n_rotate > 0:
            self.register_module('pe', RotaryPosEncoding(self.n_rotate, seq_dim=-2, base=rope_base))

    def inner_attend(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        if self.n_rotate < self.h_prim:
            r_k = K[..., :self.n_rotate]
            nr_k = K[..., self.n_rotate:]
            r_q = Q[..., :self.n_rotate]
            nr_q = Q[..., self.n_rotate:]

            r_q, r_k = self.pe(r_q, r_k)
            Q, K = torch.cat([r_q, nr_q], dim=-1), torch.cat([r_k, nr_k], dim=-1)
        else:
            Q, K = self.pe(Q, K)

        AV = F.scaled_dot_product_attention( # unsqueezes to make mask head specifc
            Q, K, V, is_causal=True
        )
        return AV
    
class MHLocalAttention(MultiHeadAttention):
    def __init__(self, h: int, h_prim: int, n_heads: int, k: int):
        super().__init__(h, h_prim, n_heads)
        from local_attention import LocalAttention
        self.local_attention = LocalAttention(dim=h_prim, window_size=int(k), causal=True)

    def inner_attend(self, Q, K, V):
        # rope encoding is included in the LocalAttention module
        mask = torch.ones(Q.shape[0], Q.shape[2], device=Q.device).bool()
        return self.local_attention(Q, K, V, mask=mask)

class MoSA(nn.Module):
    def __init__(self, 
                h: int, h_prim: int, num_mosa_heads: int, num_other_heads: int, 
                max_seq_len: int, sparsity: int,
                hybrid_type: str ='dense', include_first: int = 0,
                rotate_fraction: float = 0.5, rope_base: float=10000
                ):
        super().__init__()

        if num_mosa_heads > 0:
            self.mosa_heads = PureMoSA(num_mosa_heads, sparsity, h, h_prim, include_first,
                                   rotate_fraction, rope_base)
        else:
            self.mosa_heads = lambda x: 0
        if num_other_heads > 0:
            if hybrid_type == 'dense':
                self.other_heads = Dense(h, h_prim, num_other_heads, rotate_fraction, rope_base)
            elif hybrid_type == 'local':
                k = max_seq_len // sparsity
                self.other_heads = MHLocalAttention(h, h_prim, num_other_heads, k)
            else:
                raise Exception(f'hybrid type {hybrid_type} not recognized')
        else:
            self.other_heads = lambda x: 0

    def forward(self, X: torch.Tensor):
        return self.mosa_heads(X) + self.other_heads(X)

        

