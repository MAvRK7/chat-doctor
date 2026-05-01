import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
        

class MoELayer(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts=4, k=2, aux_loss_weight=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.aux_loss_weight = aux_loss_weight
        
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpert(dim, hidden_dim) for _ in range(num_experts)])

    def forward(self, x):
        b, s, d = x.shape
        gate_logits = self.gate(x)                    # (b, s, E)
        
        # Softmax + top-k
        gate_scores = F.softmax(gate_logits, dim=-1)
        topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=-1)

        # --- Auxiliary Losses ---
        # 1. Load balancing loss (you already had something similar)
        expert_usage = gate_scores.mean(dim=(0, 1))
        load_balance_loss = -(expert_usage * torch.log(expert_usage + 1e-9)).sum()

        # 2. Z-loss (very important for MoE stability)
        z_loss = torch.mean(torch.square(torch.logsumexp(gate_logits, dim=-1))) * 0.001

        aux_loss = load_balance_loss + z_loss

        # Dispatch + combine (your current logic is mostly fine, but fix weighting)
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = (topk_indices == e).any(dim=-1)   # (b, s)
            if not mask.any():
                continue
            tokens = x[mask]
            expert_out = self.experts[e](tokens)
            
            # Proper weighting
            weights = topk_scores[mask].sum(dim=-1, keepdim=True)  # combine top-k if needed
            expert_out = expert_out * weights
            
            out[mask] += expert_out

        return out, aux_loss, gate_scores
