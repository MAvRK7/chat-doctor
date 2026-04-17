import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.moe import MoELayer


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, t, d = x.shape
        if mask is None:
            mask = torch.ones(b, t, device=x.device)
        qkv = self.qkv(x)  # (b, t, 3d)
        q, k, v = qkv.chunk(3, dim=-1)


        def reshape(h):
            return h.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (b, h, t, hd)

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (b, h, t, t)

        # causal mask
        causal = torch.tril(torch.ones(t, t, device=x.device)).unsqueeze(0).unsqueeze(0)
        att = att.masked_fill(causal == 0, float("-inf"))

        if mask is not None:
            # mask: (batch, seq)
            # convert to (batch, 1, 1, seq)
            mask = mask[:, None, None, :]
            att = att.masked_fill(mask == 0, float("-inf"))


        att = F.softmax(att, dim=-1)
        out = att @ v  # (b, h, t, hd)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_hidden_dim, num_experts=4, k=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.moe = MoELayer(dim, ffn_hidden_dim, num_experts=num_experts, k=k)

    def forward(self, x, mask=None):
        # Self-attention
        h = self.ln1(x)
        x = x + self.attn(h, mask=mask)

        # MoE FFN
        h = self.ln2(x)
        moe_out, moe_loss = self.moe(h)
        x = x + moe_out

        return x, moe_loss


class MoETransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim=512,
        num_layers=8,
        num_heads=8,
        ffn_hidden_dim=2048,
        num_experts=4,
        k=2,
        max_seq_len=256,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ffn_hidden_dim, num_experts, k)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        b, t = input_ids.shape
        device = input_ids.device

        pos = torch.arange(0, t, device=device).unsqueeze(0)  # (1, t)

        x = self.token_emb(input_ids) + self.pos_emb(pos)

        total_moe_loss = 0.0
        for block in self.blocks:
            x, moe_loss = block(x, mask=attention_mask)
            total_moe_loss = total_moe_loss + moe_loss

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits, total_moe_loss / len(self.blocks)
