"""Retrieval-head scorer.

For a set of attention heads (layer l, head h) of the backbone, the score of candidate c is the
attention mass that the query rows put on c's tokens, averaged over the query rows (Eq. 1), and the
final score is the mean over the selected heads (Eq. 2). In RE3Rank the query rows are the
generator's reasoning trace, an open <think> block placed after the user turn; without a trace they
are the question tokens (the QRHead setting, used for the untrained baseline).

The trunk is the backbone truncated below the topmost scoring layer. For each scoring layer only its
input layernorm and Q/K path are used (ScoreHead); scoring layers below the top share these modules
with the trunk. The attention of the selected heads is recomputed explicitly in fp32.
"""
import json
import os

import torch
import torch.nn as nn
from transformers import AutoModel

from . import backbone as bbk
from .attention import get_attn_weights
from .prompts import RR_SYSTEM, build_prompt

SCALE = 8.0   # range of the max-min normalised scores inside the ranking loss


class ScoreHead(nn.Module):
    """Input layernorm + Q/K path of one scoring layer.

    k_layer (Gemma-4 E-models only): a KV-shared scoring layer takes its K path (input_layernorm /
    k_proj / k_norm) from its donor layer; the extra k_input_layernorm key exists only in that case,
    so Qwen checkpoints keep their exact key set."""

    def __init__(self, layer, config, k_layer=None, fam="qwen"):
        super().__init__()
        self.input_layernorm = layer.input_layernorm
        attn = layer.self_attn
        kl = layer if k_layer is None else k_layer
        self.q_proj, self.k_proj = attn.q_proj, kl.self_attn.k_proj
        self.q_norm, self.k_norm = attn.q_norm, kl.self_attn.k_norm
        self.k_input_layernorm = None if k_layer is None else kl.input_layernorm
        self.head_dim = attn.head_dim
        self.n_q = config.num_attention_heads
        self.n_kv = bbk.n_kv_heads(attn, config)
        self.scale = bbk.attn_scale(attn, fam)      # None = 1/sqrt(head_dim)
        self.layer_type = bbk.layer_type(attn)
        self.fam = fam


def parse_heads(spec):
    """'23-10,21-8' -> [(23, 10), (21, 8)]  (layer-head, 0-based)"""
    return [tuple(int(x) for x in t.split("-")) for t in spec.split(",")]


class HeadScorer(nn.Module):
    """Scores the candidates of one prompt with the mean attention mass of the given heads."""

    def __init__(self, model_id, head_list):
        super().__init__()
        self.fam = bbk.family(model_id)
        full = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
        full = bbk.text_model(full)     # multimodal wrapper -> its text decoder
        layers = sorted({l for l, _ in head_list})
        self.top = layers[-1]
        # row order: grouped by layer, original in-layer order preserved
        self.rows = [(l, h) for l in layers for ll, h in head_list if ll == l]
        for l in layers:
            assert bbk.is_full(full, l), f"layer {l} is a sliding-window layer"
        self.k_layer = {l: bbk.kv_donor(full, l) for l in layers}
        self.score_heads = nn.ModuleDict(
            {str(l): ScoreHead(full.layers[l], full.config,
                               None if self.k_layer[l] == l else full.layers[self.k_layer[l]],
                               self.fam) for l in layers})
        full.layers = nn.ModuleList(list(full.layers[: self.top]))
        if self.fam == "qwen":
            full.config.num_hidden_layers = self.top   # Gemma keeps it: its PLE table is reshaped by it
        full.norm = nn.Identity()
        full.config.use_cache = False
        self.trunk = full
        self.embed_scale = bbk.embed_scale(full)   # Gemma: sqrt(hidden) inside embed_tokens
        self.ple = bbk.ple_dim(full)               # Gemma E-models: per-layer embeddings
        # capture the inputs of below-top scoring layers with pre-hooks: these are checkpoint-boundary
        # tensors (kept by autograd anyway), so this costs no extra memory
        self._cap = {}
        cap_layers = sorted(set(layers[:-1]) | {d for d in self.k_layer.values() if d < self.top})
        for l in cap_layers:
            def make_hook(li):
                def hook(module, hook_args):
                    self._cap[li] = hook_args[0]
                return hook
            self.trunk.layers[l].register_forward_pre_hook(make_hook(l))

    def forward(self, input_ids, q_lo=None, q_hi=None, chunk_ranges=None,
                inputs_embeds=None, spans=None, ple_ids=None):
        """Returns (scores[C] fp32, per_head[K, C] in self.rows order).

        inputs_embeds (optional, differentiable path): bypass embed_tokens so that a straight-through
        token mixture can carry gradient into the caller; input_ids is ignored when it is given.

        spans (optional, batched path): the input is (B, Lmax) RIGHT-padded and spans =
        [(q_lo, q_hi, chunk_ranges)] per row; returns a list of (scores, per_head), one per row.
        Right padding needs no mask: causal attention keeps every real token's hidden state
        identical to the unpadded run, and positions are unchanged."""
        ref = input_ids if inputs_embeds is None else inputs_embeds
        L = ref.shape[1]
        kw = {}
        if inputs_embeds is not None and self.ple:
            # Gemma E-models: the token-identity half of the per-layer embeddings comes from the
            # hard ids (no gradient); the soft mixture only enters through the main embedding
            assert ple_ids is not None, "inputs_embeds on a PLE backbone needs ple_ids"
            with torch.no_grad():
                kw["per_layer_inputs"] = self.trunk.get_per_layer_inputs(ple_ids, None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = (self.trunk(input_ids=input_ids) if inputs_embeds is None
                   else self.trunk(inputs_embeds=inputs_embeds, **kw))
        hs = dict(self._cap)  # hs[l] = input to layer l
        self._cap.clear()
        hs[self.top] = out.last_hidden_state  # norm = Identity -> input to the top layer
        position_ids = torch.arange(L, device=ref.device).unsqueeze(0)
        pe = {lt: bbk.rope_cos_sin(self.trunk, hs[self.top].float(), position_ids, lt)
              for lt in sorted({sh.layer_type for sh in self.score_heads.values()})}
        if spans is not None:
            return [self._score(hs, pe, b, lo, hi, cr)
                    for b, (lo, hi, cr) in enumerate(spans)]
        return self._score(hs, pe, 0, q_lo, q_hi, chunk_ranges)

    def _score(self, hs, pe, b, q_lo, q_hi, chunk_ranges):
        upper = q_hi
        per_head = []
        for l in sorted({l for l, _ in self.rows}):
            sh = self.score_heads[str(l)]
            cos, sin = pe[sh.layer_type]
            x = sh.input_layernorm(hs[l][b:b + 1, :upper].float())
            q = sh.q_norm(sh.q_proj(x[:, q_lo:q_hi]).view(
                1, q_hi - q_lo, sh.n_q, sh.head_dim)).transpose(1, 2)
            xk = (x if sh.k_input_layernorm is None
                  else sh.k_input_layernorm(hs[self.k_layer[l]][b:b + 1, :upper].float()))
            k = sh.k_norm(sh.k_proj(xk[:, :upper]).view(
                1, upper, sh.n_kv, sh.head_dim)).transpose(1, 2)
            heads_l = [h for ll, h in self.rows if ll == l]
            # only the scored heads go through the explicit fp32 attention: softmax is per head, so
            # slicing q to heads_l and k to their GQA rows is exact
            grp = sh.n_q // sh.n_kv
            q = q[:, heads_l]
            k = k[:, [h // grp for h in heads_l]]
            q = bbk.apply_rope(sh.fam, q, cos[:, q_lo:q_hi], sin[:, q_lo:q_hi])
            k = bbk.apply_rope(sh.fam, k, cos[:, :upper], sin[:, :upper])
            aw = get_attn_weights(k, q, sh.scale).mean(dim=-2)  # (1, Hl, upper)
            per_head.append(torch.stack(
                [aw[0, :, s:e].sum(dim=-1) for s, e in chunk_ranges], dim=1))
        per_head = torch.cat(per_head, dim=0)  # (K, C)
        return per_head.mean(dim=0), per_head


def load_scorer(model_id, heads, ckpt=""):
    """HeadScorer on the backbone, with trained weights loaded when given (else untrained): the
    scorer.pt of a training run, or the scorer/scorer.safetensors of a released model, whose frozen
    tables listed in scorer/config.json ("from_base") keep the values loaded from the base model."""
    net = HeadScorer(model_id, parse_heads(heads))
    if ckpt.endswith(".safetensors"):
        from safetensors.torch import load_file
        with open(os.path.join(os.path.dirname(ckpt), "config.json")) as f:
            cfg = json.load(f)
        assert parse_heads(cfg["heads"]) == parse_heads(heads), f"{ckpt} was trained with heads {cfg['heads']}"
        sd = load_file(ckpt)
        missing, unexpected = net.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
        assert not unexpected and set(missing) <= set(cfg["from_base"]), (missing[:5], unexpected[:5])
        print(f"loaded {ckpt} ({len(sd)} tensors; from the base model: {', '.join(missing) or 'none'})")
    elif ckpt:
        sd = torch.load(ckpt, map_location="cpu")
        missing, unexpected = net.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
        print(f"loaded {ckpt} (missing {len(missing)}, unexpected {len(unexpected)})")
    return net


def build_instance(item, tokenizer, model_id, trace=None, max_len=32768):
    """Scorer input for one item.

    With a trace: [system][user turn][assistant opener][think opener][trace] (open block, no closing
    tag), query rows = the trace tokens. Without one: the bare user turn, query rows = the question
    tokens. Returns input_ids (1, L), the token range of every candidate, the query row range and
    the gold mask."""
    f = bbk.fmt_for(model_id)
    full_seq, chunk_ranges = build_prompt(item["question"], item["paragraphs"], f.user_open)
    if trace:
        pre = f.sys_pre(RR_SYSTEM)
        full_seq = pre + full_seq + f"{f.assistant_open}{f.think_open}{trace}"
        chunk_ranges = [[a + len(pre), b + len(pre)] for a, b in chunk_ranges]
    enc = tokenizer(full_seq, max_length=max_len, truncation=True,
                    return_tensors="pt", return_offsets_mapping=True,
                    add_special_tokens=False)
    offsets = enc["offset_mapping"][0].tolist()
    char_to_token = {}
    for i, (s, e) in enumerate(offsets):
        for j in range(s, e):
            char_to_token[j] = i
    token_chunk_ranges = [
        [char_to_token.get(s, 0), char_to_token.get(e - 1, 0) + 1]
        for s, e in chunk_ranges
    ]
    q = trace if trace else item["question"]
    q_start = full_seq.rindex(q)
    if trace:
        # strict containment: only tokens fully inside the trace span are query rows
        inside = [i for i, (a, b) in enumerate(offsets)
                  if a >= q_start and b <= q_start + len(q) and b > a]
        if inside:
            q_lo, q_hi = inside[0], inside[-1] + 1
        else:   # a trace so short that both boundary tokens straddle it
            q_lo = char_to_token.get(q_start, 0)
            q_hi = char_to_token.get(q_start + len(q) - 1, q_lo) + 1
    else:
        q_lo = char_to_token[q_start]
        q_hi = char_to_token[q_start + len(q) - 1] + 1
    return {
        "input_ids": enc["input_ids"],
        "chunk_ranges": token_chunk_ranges,
        "q_lo": q_lo,
        "q_hi": q_hi,
        "gold": torch.tensor([bool(p.get("is_supporting")) for p in item["paragraphs"]]),
    }


def ranking_loss(scores, gold, loss="ce+rsatk", ks=(3, 10), tau1=1.0, tau2=1.0, mix=2.0, scale=SCALE):
    """Ranking loss on one item's candidate scores (gold = bool mask, at least one gold and one
    negative). Scores are first max-min normalised to [0, scale].

    ce        grouped contrastive loss: -mean_p log( e^{s_p} / (e^{s_p} + sum_n e^{s_n}) )
    rsatk     Recall@k surrogate (Patel et al., CVPR 2022): both step functions of recall@k replaced
              by sigmoids (score axis tau2, rank axis tau1), averaged over k in ks with the
              min(k, |G|) multi-gold denominator
    ce+rsatk  ce + mix * rsatk   (the RE3Rank recipe: ks = (3, 10), tau1 = tau2 = 1, mix = 2)"""
    if loss == "ce+rsatk":
        return (ranking_loss(scores, gold, "ce", scale=scale)
                + mix * ranking_loss(scores, gold, "rsatk", ks, tau1, tau2, scale=scale))
    s = scale * (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    pos, neg = s[gold], s[~gold]
    if loss == "rsatk":
        diff = (s.unsqueeze(0) - pos.unsqueeze(1)) / tau2  # [p, z] = s_z - s_p
        ranks = 1.0 + torch.sigmoid(diff).sum(1) - 0.5     # -0.5: the z == p self term
        g = ranks.numel()
        terms = []
        for kk in ks:
            denom = float(min(kk, g))
            hits = torch.sigmoid((kk + 0.0 - ranks) / tau1).sum().clamp(max=denom)
            terms.append(1.0 - hits / denom)
        return torch.stack(terms).mean()
    assert loss == "ce", loss
    neg_lse = torch.logsumexp(neg, dim=0)
    return -(pos - torch.logaddexp(pos, neg_lse.expand_as(pos))).mean()
