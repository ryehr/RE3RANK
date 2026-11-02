"""Explicit causal attention weights for a block of query rows (the scorer recomputes the
attention of its selected heads in fp32 instead of reading it from a fused kernel)."""
import math

import torch


def repeat_kv(hidden_states, n_rep):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def get_attn_weights(key_states, query_states, scale=None):
    """Softmax attention of query_states (B, H, Lq, d) over key_states (B, Hkv, Lk, d), where the
    query rows are the last Lq positions of the Lk keys (causal mask). scale=None means
    1/sqrt(head_dim) (Qwen); Gemma-4 passes its own."""
    bsz, num_heads, q_len, head_dim = query_states.size()
    num_kv_heads = key_states.size(1)
    key_states = repeat_kv(key_states, num_heads // num_kv_heads)

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    attn_weights = torch.matmul(query_states * scale, key_states.transpose(2, 3))

    seq_len = attn_weights.size(-1)
    causal_mask = torch.ones(num_heads, q_len, seq_len, device=attn_weights.device)
    causal_mask = torch.triu(causal_mask.transpose(-1, -2), diagonal=-(seq_len - q_len)).transpose(-1, -2)
    attn_weights += ((1 - causal_mask) * torch.finfo(attn_weights.dtype).min).unsqueeze(0)

    attn_lses = torch.logsumexp(attn_weights, dim=-1, keepdim=True)
    return torch.exp(attn_weights - attn_lses)
