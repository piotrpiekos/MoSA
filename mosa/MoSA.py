import math

import torch
from torch import nn
from torch.nn import functional as F

from .positional_encoding import MoSARotaryPosEncoding

class ExpertGather(nn.Module):
    def __init__(self, E: int, I: int, J: int):
        super().__init__()

        self.E, self.I, self.J = E, I, J
        self.W = nn.Parameter(torch.empty(E, I, J))

        self.reset_parameters()

    def forward(self, X: torch.Tensor, ind: torch.Tensor):
        # output has shape [B,E,K,J]
        B, T, I = X.shape
        _, E, K = ind.shape
        

        index=ind.reshape(B, E*K)[...,None].expand(-1,-1,I)
        X_gathered = torch.gather(X, dim=1, index=index).reshape(B, E, K, I)
        Y = torch.einsum('beki, eij->bekj', X_gathered, self.W)
        return Y
    
    def reset_parameters(self):
        bound = 1 / math.sqrt(self.J)
        nn.init.uniform_(self.W, -bound, bound)


class ExpertScatter(nn.Module):
    def __init__(self, E: int, J: int, I: int):
        super().__init__()

        self.E, self.J, self.I = E, J, I
        self.W = nn.Linear(J*E, I, bias=False)


    def forward(self, Y: torch.Tensor, Ind: torch.Tensor, T: int):
        B, E, K, J = Y.shape

        scattered = torch.zeros(B, E, T, J, device=Y.device, dtype=Y.dtype)
        unfolded_out = torch.scatter(scattered, dim=2,
                                     index=Ind[..., None].expand(-1,-1,-1,J),
                                     src=Y) # [B, E, T, J]
        out = self.W(unfolded_out.transpose(1,2).reshape(B, T, -1)) #[b, seq_len, h]
        return out
        


class ExpertScatter(nn.Module):
    def __init__(self, E: int, J: int, I: int):
        super().__init__()

        self.E, self.J, self.I = E, J, I
        self.W = nn.Parameter(torch.empty(E, J, I))

        self.reset_parameters()

    def forward(self, Y: torch.Tensor, Ind: torch.Tensor, T: int):
        B, E, K, J = Y.shape
        # Ind shape [B, E, K]

        X_prescatter = torch.einsum('bekj, eji->beki', Y, self.W)

        I = X_prescatter.shape[-1]
        scattered = torch.zeros(B, T, I, device=Y.device, dtype=Y.dtype)
        Ind = Ind[..., None].expand(-1,-1,-1,I)
        scattered.scatter_add_(1, Ind.reshape(B, E*K, I), X_prescatter.reshape(B, E*K, I))
        return scattered
    
    def reset_parameters(self):
        bound = 1 / math.sqrt(self.I*self.E)
        nn.init.uniform_(self.W, -bound, bound)




class PureMoSA(nn.Module):
    def __init__(self, n_heads: int, sparsity: int, h: int, h_prim: int, include_first: int = 0, 
                 rotate_fraction: float = 0.5, rope_base: float = 10000, 
                 ):
        super().__init__()

        self.n_heads, self.h, self.h_prim, self.include_first = n_heads, h, h_prim, include_first
        self.sparsity = sparsity

        self.r = torch.nn.Sequential(
            torch.nn.Linear(h, n_heads, bias=False),
            torch.nn.Sigmoid()
        )

        self.QKV = ExpertGather(n_heads, h, 3*h_prim)
        self.O = ExpertScatter(n_heads, h_prim, h)

        # rope
        self.n_rotate = int(rotate_fraction * self.h_prim)
        self.n_rotate -= self.n_rotate % 2
        if self.n_rotate > 0:
            self.register_module('pe', MoSARotaryPosEncoding(self.n_rotate, seq_dim=-2, base=rope_base))


    def get_topk_includefirst(self, logits: torch.Tensor):
        # logits: [B, T, E]
        B, T, E = logits.shape
        k = int(T // self.sparsity)
        k = min(max(k, 2), T)           # at least 2, at most T
        k1 = k - 1                      
        
        tail_vals, tail_idx = torch.topk(logits[:, 1:, :], k=k1,dim=1)
        # tail_vals: [B, k1, E], tail_idx: [B, k1, E]

        first_vals = logits[:, :1, :]   # [B, 1, E]
        first_idx  = torch.zeros(B, 1, E, dtype=torch.long, device=logits.device)

        tail_idx = tail_idx + 1         # now in [1 … T-1]

        vals = torch.cat([first_vals, tail_vals], dim=1)  # [B, k, E]
        idxs = torch.cat([first_idx,  tail_idx ], dim=1)  # [B, k, E]

        return vals.transpose(1,2), idxs.transpose(1,2), k

    def get_topk(self, x: torch.Tensor):
        """
        Selects tokens for the experts
        Input:
            x - inputs shape [B, T, E]
        Output (3-tuple):
            - scores of the tokens for given router [B, E, k]
            - indices of selected tokens in the original sequence [B, E, k]
            - selected number of tokens
        """
        B, T, h = x.shape

        logits = self.r(x)

        if self.include_first:
            return self.get_topk_includefirst(logits)

        k = int(T // self.sparsity)
        k = min(max(k, 2), T) # 2 is the minimum number of tokens to select from

        logits_topk = logits.topk(dim=1, k=k) # [b, k, E]
        topk_I = logits_topk.indices.transpose(1,2) # [b, E, k]
        topk_vals = logits_topk.values.transpose(1,2) # [b, E, k]

        return topk_vals, topk_I, k
    
    def inner_attend(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, topk_I: torch.Tensor):
        M = topk_I.unsqueeze(-1) >= topk_I.unsqueeze(-2)
        if self.n_rotate < self.h_prim:
            r_k = K[..., :self.n_rotate]
            nr_k = K[..., self.n_rotate:]
            r_q = Q[..., :self.n_rotate]
            nr_q = Q[..., self.n_rotate:]

            r_q, r_k = self.pe(r_q, topk_I, r_k, topk_I)
            Q, K = torch.cat([r_q, nr_q], dim=-1), torch.cat([r_k, nr_k], dim=-1)
        else:
            Q, K = self.pe(Q, topk_I, K, topk_I)

        AV = F.scaled_dot_product_attention( # unsqueezes to make mask head specifc
            Q.unsqueeze(2), K.unsqueeze(2), V.unsqueeze(2),
            attn_mask=M.bool().unsqueeze(2)
        ).squeeze(2)

        return AV
    
    def forward(self, X: torch.Tensor):
        # X has shape [b, T, h]
        b, T, _ = X.shape

        topk_vals, topk_I, k = self.get_topk(X)

        Q, K, V = self.QKV(X, topk_I).chunk(3, dim=-1) # [B, E, k, self.h_prim]
   
        AV = self.inner_attend(Q, K, V, topk_I)
        AV = AV * topk_vals.unsqueeze(-1) # [B, E, k, h_prim]

        return self.O(AV, topk_I, T)


