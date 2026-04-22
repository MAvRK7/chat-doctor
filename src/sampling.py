import torch
import torch.nn.functional as F

def top_k_top_p_filtering(logits, top_k=50, top_p=0.9):
    # Top-k
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1)
        logits = torch.where(logits < min_values, torch.full_like(logits, -1e10), logits)

    # Top-p (nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        mask = cumulative_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False

        sorted_logits[mask] = -1e10
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

    return logits

def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    # logits: (1, vocab_size)
    # generated_ids: list of token IDs already generated
    for token_id in set(generated_ids):
        logits[0, token_id] /= penalty
    return logits



def sample(logits, temperature=1.0, top_k=50, top_p=0.9):
    logits = logits / temperature
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
