"""Backbone adapter for the head scorer and the joint trainer.

Two families: "qwen" (Qwen3, the models used in the paper) and "gemma4"
(google/gemma-4-*-it: a multimodal wrapper around a text decoder with interleaved
sliding/full attention, KV-shared top layers on the E-models, per-layer-type RoPE,
scaled embeddings and a native reasoning channel instead of <think> tags).
"""
import inspect
import math

import torch


def family(x):
    """'qwen' | 'gemma4' from a model id / path / config."""
    if not isinstance(x, str):
        x = getattr(x, "model_type", "") or ""
    x = x.lower()
    return "gemma4" if ("gemma-4" in x or "gemma4" in x) else "qwen"


class Fmt:
    """Chat literals for the scorer-side prompt builders. They must reproduce
    tok.apply_chat_template([system, user], add_generation_prompt=True,
    enable_thinking=True) byte for byte, followed by the forced-open
    reasoning marker the generator is trained to continue from."""

    def __init__(self, fam):
        self.fam = fam
        if fam == "gemma4":
            self._sys = "<bos><|turn>system\n<|think|>\n{}<turn|>\n"
            self.user_open = "<|turn>user\n"
            self.assistant_open = "<turn|>\n<|turn>model\n"
            self.think_open = "<|channel>thought\n"
            self.think_close = "<channel|>"
            self.extra_stop = ["<turn|>"]
        else:
            self._sys = "<|im_start|>system\n{}<|im_end|>\n"
            self.user_open = "<|im_start|>user\n"
            self.assistant_open = "<|im_end|>\n<|im_start|>assistant\n"
            self.think_open = "<think>"
            self.think_close = "</think>"
            self.extra_stop = []

    def sys_pre(self, system):
        return self._sys.format(system)

    def think_open_ids(self, tok):
        return tok.encode(self.think_open, add_special_tokens=False)

    def think_close_id(self, tok):
        ids = tok.encode(self.think_close, add_special_tokens=False)
        assert len(ids) == 1, (self.think_close, ids)
        return ids[0]

    def stop_ids(self, tok):
        eos = tok.eos_token_id
        ids = [eos, self.think_close_id(tok)]
        for s in self.extra_stop:
            t = tok.encode(s, add_special_tokens=False)
            if len(t) == 1 and t[0] not in ids:
                ids.append(t[0])
        return ids


def fmt_for(model_id):
    return Fmt(family(model_id))


# ---------------------------------------------------------------- model access

def text_model(m):
    """The decoder stack (.layers / .embed_tokens / .norm / .rotary_emb) behind
    an AutoModel / AutoModelForCausalLM result (multimodal wrappers nest it)."""
    if hasattr(m, "language_model"):
        return m.language_model
    if hasattr(m, "model") and hasattr(m.model, "language_model"):
        return m.model.language_model
    return m


def freeze_non_text(lm):
    """Multimodal towers of a generator wrapper: never trained, never used
    (text-only prompts). Freezing keeps DDP/optimizer off them."""
    n = 0
    for name, p in lm.named_parameters():
        if any(t in name for t in ("vision_tower", "audio_tower",
                                   "embed_vision", "embed_audio")):
            p.requires_grad_(False)
            n += p.numel()
    return n


def freeze_ple(lm):
    """Gemma-4 E-models: the per-layer-embedding token table (V x L*256, 2.3-2.8B
    entries) exceeds the int32 element limit of the bitsandbytes 8-bit optimizer
    kernels ('invalid argument at ops.cu'), so it stays frozen in the generator
    (the scorer freezes it anyway, like embed_tokens)."""
    n = 0
    for name, p in lm.named_parameters():
        if "embed_tokens_per_layer" in name:
            p.requires_grad_(False)
            n += p.numel()
    return n


def layer_type(attn):
    return getattr(attn, "layer_type", None) or "full_attention"


def is_full(tm, li):
    return layer_type(tm.layers[li].self_attn) == "full_attention"


def kv_donor(tm, li):
    """Layer whose K/V feed layer li's attention: li itself, or (Gemma-4
    E-models, KV-shared top layers) the last non-shared layer of the same
    attention type (the one flagged store_full_length_kv)."""
    a = tm.layers[li].self_attn
    if not getattr(a, "is_kv_shared_layer", False):
        return li
    lt = layer_type(a)
    for j in range(li - 1, -1, -1):
        b = tm.layers[j].self_attn
        if getattr(b, "store_full_length_kv", False) and layer_type(b) == lt:
            return j
    raise ValueError(f"no KV donor for shared layer {li}")


def attn_scale(attn, fam):
    """Softmax scale. None = the historical 1/sqrt(head_dim) formula (Qwen)."""
    if fam == "qwen":
        return None
    return float(getattr(attn, "scaling", attn.head_dim ** -0.5))


def n_kv_heads(attn, config):
    grp = getattr(attn, "num_key_value_groups", None)
    if grp:
        return config.num_attention_heads // grp
    return config.num_key_value_heads


_ROPE_HAS_TYPE = {}


def rope_cos_sin(tm, x, position_ids, lt="full_attention"):
    re = tm.rotary_emb
    key = id(re)
    if key not in _ROPE_HAS_TYPE:
        _ROPE_HAS_TYPE[key] = "layer_type" in inspect.signature(re.forward).parameters
    if _ROPE_HAS_TYPE[key]:
        return re(x, position_ids, lt)
    return re(x, position_ids)


def apply_rope(fam, x, cos, sin):
    """RoPE on a (B, H, L, hd) tensor; cos/sin (B, L, rot) as the model's own
    rotary module produced them for this layer type."""
    if fam == "gemma4":
        from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb as ar
        return ar(x, cos, sin)
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as ar
    return ar(x, x, cos, sin)[0]


def embed_scale(tm):
    e = tm.embed_tokens
    return float(getattr(e, "scalar_embed_scale", 1.0))


def ple_dim(tm):
    return int(getattr(tm, "hidden_size_per_layer_input", 0) or 0)


def copy_aux_files(base_id, dst):
    """Processor / preprocessor configs a multimodal wrapper checkpoint needs
    (vLLM builds the mm processor even for text-only prompts)."""
    import os
    import shutil
    from huggingface_hub import try_to_load_from_cache
    out = []
    for fn in ("processor_config.json", "preprocessor_config.json",
               "chat_template.jinja", "generation_config.json"):
        if os.path.isfile(os.path.join(dst, fn)):
            continue
        p = try_to_load_from_cache(base_id, fn)
        if isinstance(p, str) and os.path.isfile(p):
            shutil.copy(p, os.path.join(dst, fn))
            out.append(fn)
    out += fill_dropped_weights(base_id, dst)
    return out


def fill_dropped_weights(base_id, dst):
    """transformers builds no k_proj/v_proj/k_norm on Gemma-4 KV-shared layers,
    so save_pretrained drops them; vLLM's loader still requires every base key.
    Copy the missing tensors from the base checkpoint into an extra shard
    (vLLM reads every *.safetensors in the dir; transformers reads
    model.safetensors alone, so training resume is unaffected)."""
    import glob
    import os
    from huggingface_hub import try_to_load_from_cache
    from safetensors import safe_open
    from safetensors.torch import save_file
    extra = os.path.join(dst, "model-dropped.safetensors")
    if os.path.isfile(extra):
        return []
    have = set()
    for f in glob.glob(os.path.join(dst, "*.safetensors")):
        with safe_open(f, "pt") as s:
            have |= set(s.keys())
    if not have:
        return []
    cfg = try_to_load_from_cache(base_id, "config.json")
    if not isinstance(cfg, str):
        return []
    fill = {}
    for f in glob.glob(os.path.join(os.path.dirname(cfg), "*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if k not in have:
                    fill[k] = s.get_tensor(k)
    if not fill:
        return []
    save_file(fill, extra, metadata={"format": "pt"})
    return [f"model-dropped.safetensors({len(fill)})"]
