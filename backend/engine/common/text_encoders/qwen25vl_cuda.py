"""Qwen2.5-VL 文本编码 — HF / PyTorch 路径（形态 B：与 ``qwen25vl_mlx`` 分离）。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

_QWEN25_PROMPT_ENCODE_PREFIX = (
    "<|im_start|>system\nAs an image captioning expert, generate a descriptive text prompt "
    "based on an image content, suitable for input to a text-to-image model.<|im_end|>\n<|im_start|>user\n"
)
_QWEN25_PROMPT_ENCODE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def _split_quotation(prompt: str, quote_pairs=None):
    word_internal_quote_pattern = re.compile(r"[a-zA-Z]+'[a-zA-Z]+")
    matches_word_internal_quote_pattern = word_internal_quote_pattern.findall(prompt)
    mapping_word_internal_quote = []

    for i, word_src in enumerate(set(matches_word_internal_quote_pattern)):
        word_tgt = "qwen25_$##$_tok" * (i + 1)
        prompt = prompt.replace(word_src, word_tgt)
        mapping_word_internal_quote.append([word_src, word_tgt])

    if quote_pairs is None:
        quote_pairs = [("'", "'"), ('"', '"'), ("‘", "’"), ("“", "”")]
    pattern = "|".join(
        [re.escape(q1) + r"[^" + re.escape(q1 + q2) + r"]*?" + re.escape(q2) for q1, q2 in quote_pairs]
    )
    parts = re.split(f"({pattern})", prompt)

    result = []
    for part in parts:
        for word_src, word_tgt in mapping_word_internal_quote:
            part = part.replace(word_tgt, word_src)
        if re.match(pattern, part):
            if len(part):
                result.append((part, True))
        else:
            if len(part):
                result.append((part, False))
    return result


def encode_prompt_embeds_qwen25vl_numpy(
    tokenizer,
    model,
    prompts: list[str],
    *,
    tokenizer_max_length: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """``[B, tokenizer_max_length, hidden]`` float32 numpy — MLX 侧再 ``mx.array``。"""
    import torch

    batch_all_tokens = []
    for each_prompt in prompts:
        all_tokens: list[int] = []
        for clean_prompt_sub, matched in _split_quotation(each_prompt):
            if matched:
                for sub_word in clean_prompt_sub:
                    tokens = tokenizer(sub_word, add_special_tokens=False)["input_ids"]
                    all_tokens.extend(tokens)
            else:
                tokens = tokenizer(clean_prompt_sub, add_special_tokens=False)["input_ids"]
                all_tokens.extend(tokens)

        if len(all_tokens) > tokenizer_max_length:
            all_tokens = all_tokens[:tokenizer_max_length]
        batch_all_tokens.append(all_tokens)

    text_tokens_and_mask = tokenizer.pad(
        {"input_ids": batch_all_tokens},
        max_length=tokenizer_max_length,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )

    prefix_tokens = tokenizer(_QWEN25_PROMPT_ENCODE_PREFIX, add_special_tokens=False)["input_ids"]
    suffix_tokens = tokenizer(_QWEN25_PROMPT_ENCODE_SUFFIX, add_special_tokens=False)["input_ids"]
    prefix_len = len(prefix_tokens)
    suffix_len = len(suffix_tokens)

    prefix_tokens_mask = torch.tensor([1] * len(prefix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype)
    suffix_tokens_mask = torch.tensor([1] * len(suffix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype)

    prefix_tokens_t = torch.tensor(prefix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)
    suffix_tokens_t = torch.tensor(suffix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)

    batch_size = text_tokens_and_mask.input_ids.size(0)
    prefix_tokens_batch = prefix_tokens_t.unsqueeze(0).expand(batch_size, -1)
    suffix_tokens_batch = suffix_tokens_t.unsqueeze(0).expand(batch_size, -1)
    prefix_mask_batch = prefix_tokens_mask.unsqueeze(0).expand(batch_size, -1)
    suffix_mask_batch = suffix_tokens_mask.unsqueeze(0).expand(batch_size, -1)

    input_ids = torch.cat((prefix_tokens_batch, text_tokens_and_mask.input_ids, suffix_tokens_batch), dim=-1)
    attention_mask = torch.cat((prefix_mask_batch, text_tokens_and_mask.attention_mask, suffix_mask_batch), dim=-1)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        prompt_embeds = outputs.hidden_states[-1][:, prefix_len:-suffix_len, :]

    return prompt_embeds.cpu().float().numpy()


def _load_filtered_safetensors_state_dict(encoder_dir: Path) -> dict[str, Any] | None:
    """Merge ``*.safetensors`` under ``encoder_dir``, dropping keys ``lm_head`` / ``lm_head.*``。"""
    try:
        from safetensors.torch import load_file
    except ImportError:
        return None

    merged: dict[str, Any] = {}
    index_fp = encoder_dir / "model.safetensors.index.json"
    if index_fp.is_file():
        meta = json.loads(index_fp.read_text(encoding="utf-8"))
        weight_map = meta.get("weight_map") or {}
        for shard in sorted(set(weight_map.values())):
            fp = encoder_dir / shard
            if not fp.is_file():
                continue
            part = load_file(str(fp))
            for k, v in part.items():
                if k == "lm_head" or k.startswith("lm_head."):
                    continue
                merged[k] = v
        return merged or None

    files = sorted(encoder_dir.glob("*.safetensors"))
    if not files:
        return None
    for fp in files:
        merged.update(load_file(str(fp)))
    merged = {
        k: v for k, v in merged.items()
        if k != "lm_head" and not k.startswith("lm_head.")
    }
    return merged or None


def load_qwen25_vl_torch_model(text_encoder_path: Path | str, device: str = "cpu"):
    """Load Qwen2.5-VL backbone for embedding extraction (no LM head required).

    Prefer loading filtered safetensors (excluding ``lm_head``) into ``Qwen2_5_VLModel``
    so checkpoints from ``ForConditionalGeneration`` match the backbone class without
    UNEXPECTED-key reports.

    Falls back to plain ``from_pretrained`` then ``AutoModel`` + ``trust_remote_code=True``.
    """
    import torch
    from transformers import AutoModel, Qwen2_5_VLModel

    path_str = str(text_encoder_path)
    path = Path(path_str)
    dev = (device or "cpu").strip().lower()
    # MPS fp16 is unstable for Qwen2.5-VL hidden-state extraction on Apple Silicon.
    dtype = torch.float32 if dev in ("mps", "cpu") else torch.float16
    kw = dict(torch_dtype=dtype, low_cpu_mem_usage=True)

    filtered_sd = _load_filtered_safetensors_state_dict(path)
    if filtered_sd:
        try:
            m = Qwen2_5_VLModel.from_pretrained(
                path_str,
                state_dict=filtered_sd,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
                trust_remote_code=False,
            )
            return m.to(device).eval()
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            pass

    try:
        m = Qwen2_5_VLModel.from_pretrained(path_str, trust_remote_code=False, **kw)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        m = AutoModel.from_pretrained(path_str, trust_remote_code=True, **kw)
    return m.to(device).eval()
