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
        gate = F.silu(self.w2(x)) # True SwiGLU
        x = self.w1(x) * gate
        return self.w3(x) * (1 / 1.41421356237)   # √2 scaling
        

class MoELayer(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts=4, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k

        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(dim, hidden_dim) for _ in range(num_experts)]
        )

    def forward(self, x):
        b, s, d = x.shape

        # Gating
        gate_logits = self.gate(x)                 # (b, s, E)
        gate_scores = F.softmax(gate_logits, -1)   # (b, s, E)

        # Top‑k routing
        topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=-1)  # (b, s, k)

        # Load‑balancing loss (entropy)
        expert_usage = gate_scores.mean(dim=(0, 1))
        load_balance_loss = -(expert_usage * torch.log(expert_usage + 1e-9)).sum()

        # Prepare output buffer
        out = torch.zeros_like(x)

        # Process each expert
        for e in range(self.num_experts):
            # mask: (b, s, k)
            mask = (topk_indices == e)

            if not mask.any():
                continue

            # Flatten mask to gather tokens
            flat_mask = mask.any(dim=-1)  # (b, s)
            tokens = x[flat_mask]         # (num_tokens, d)

            # Expert forward
            expert_out = self.experts[e](tokens)

            # Gate weights for these tokens
            weights = topk_scores[mask].unsqueeze(-1)  # (num_tokens, 1)

            # Weighted expert output
            expert_out = expert_out * weights

            # Scatter back
            out[flat_mask] += expert_out

        return out, load_balance_loss
