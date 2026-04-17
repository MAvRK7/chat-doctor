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
        gate = torch.sigmoid(self.w2(x))
        x = self.w1(x) * gate
        return self.w3(x)


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
        """
        x: (batch, seq, dim)
        """
        b, s, d = x.shape

        # Gating
        gate_logits = self.gate(x)                 # (b, s, E)
        gate_scores = F.softmax(gate_logits, -1)   # (b, s, E)

        # Top‑k routing
        topk_scores, topk_indices = torch.topk(
            gate_scores, self.k, dim=-1
        )  # (b, s, k)

        # Load‑balancing loss (encourage uniform usage)
        expert_usage = gate_scores.mean(dim=(0, 1))  # (E,)
        load_balance_loss = (expert_usage * torch.log(expert_usage + 1e-9)).sum()

        # Output buffer
        output = torch.zeros_like(x)

        # Route tokens
        for i in range(self.k):
            expert_idx = topk_indices[..., i]      # (b, s)
            expert_scores = topk_scores[..., i]    # (b, s)

            for e in range(self.num_experts):
                mask = (expert_idx == e)           # (b, s)

                if mask.any():
                    # 1. Tokens for this expert
                    tokens = x[mask]               # (num_tokens, d)

                    # 2. Forward through expert
                    expert_out = self.experts[e](tokens)  # (num_tokens, d)

                    # 3. Gate weights for these tokens
                    weights = expert_scores[mask].unsqueeze(-1)  # (num_tokens, 1)

                    # 4. Scatter back
                    output[mask] += expert_out * weights

        return output, load_balance_loss
