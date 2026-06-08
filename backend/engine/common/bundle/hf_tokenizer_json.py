"""Minimal HuggingFace ``tokenizer.json`` encoder — no ``transformers`` / ``tokenizers`` deps."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

try:
    import regex as _re_unicode
except ImportError:  # pragma: no cover - optional; Split pretokenizer needs it
    _re_unicode = None

_BYTE_TO_UNICODE: dict[int, str] | None = None
_UNICODE_TO_BYTE: dict[str, int] | None = None


def _byte_maps() -> tuple[dict[int, str], dict[str, int]]:
    global _BYTE_TO_UNICODE, _UNICODE_TO_BYTE
    if _BYTE_TO_UNICODE is not None and _UNICODE_TO_BYTE is not None:
        return _BYTE_TO_UNICODE, _UNICODE_TO_BYTE
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    out: dict[int, str] = {}
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            out[b] = chr(256 + n)
            n += 1
    byte_to_uni = {b: chr(b) for b in bs[: 256 - n]} | out
    uni_to_byte = {v: k for k, v in byte_to_uni.items()}
    _BYTE_TO_UNICODE = byte_to_uni
    _UNICODE_TO_BYTE = uni_to_byte
    return byte_to_uni, _UNICODE_TO_BYTE


def _byte_level_pretokenize(text: str, *, add_prefix_space: bool = False) -> str:
    if add_prefix_space and text and not text[0].isspace():
        text = " " + text
    b2u, _ = _byte_maps()
    return "".join(b2u[b] for b in text.encode("utf-8"))


def _split_with_pattern(text: str, pattern: str) -> list[str]:
    if _re_unicode is None:
        raise RuntimeError(
            "HFTokenizerJson Split pretokenizer requires the 'regex' package "
            "(Unicode property classes in tokenizer.json)."
        )
    compiled = _re_unicode.compile(pattern)
    i = 0
    out: list[str] = []
    while i < len(text):
        match = compiled.match(text, i)
        if match is None:
            raise RuntimeError(
                f"HFTokenizerJson Split pretokenizer stuck at index {i} "
                f"in {text[i : i + 32]!r}."
            )
        out.append(match.group())
        i = match.end()
    return out


def _split_pattern_from_node(node: dict[str, Any]) -> str:
    pattern = node.get("pattern")
    if isinstance(pattern, dict) and pattern.get("Regex"):
        return str(pattern["Regex"])
    if isinstance(pattern, str):
        return pattern
    raise RuntimeError(f"Unsupported Split pretokenizer pattern: {pattern!r}")


class HFTokenizerJson:
    """Encode-only BPE tokenizer loaded from ``tokenizer.json``."""

    def __init__(self, root: Path, data: dict[str, Any], config: dict[str, Any]):
        self.root = Path(root)
        self.config = config
        model = data.get("model") or {}
        if model.get("type") != "BPE":
            raise RuntimeError(
                f"HFTokenizerJson only supports BPE tokenizer.json (got {model.get('type')!r}) under {root}."
            )
        self.vocab: dict[str, int] = {str(k): int(v) for k, v in (model.get("vocab") or {}).items()}
        self.unk_id = int(self.vocab.get("<unk>", 0))
        merges_raw = model.get("merges") or []
        self.merge_ranks: dict[tuple[str, str], int] = {}
        for i, merge in enumerate(merges_raw):
            if isinstance(merge, list) and len(merge) == 2:
                self.merge_ranks[(str(merge[0]), str(merge[1]))] = i
            elif isinstance(merge, str):
                parts = merge.split()
                if len(parts) == 2:
                    self.merge_ranks[(parts[0], parts[1])] = i
        self._added: dict[str, int] = {}
        for item in data.get("added_tokens") or []:
            if isinstance(item, dict) and item.get("content") is not None:
                self._added[str(item["content"])] = int(item["id"])
        self._pre_tokenizer = data.get("pre_tokenizer")
        self._post_processor = data.get("post_processor")

    @classmethod
    def from_directory(cls, tokenizer_dir: Path | str) -> "HFTokenizerJson":
        root = Path(tokenizer_dir)
        tok_path = root / "tokenizer.json"
        if not tok_path.is_file():
            raise RuntimeError(f"Missing tokenizer.json under {root}")
        data = json.loads(tok_path.read_text(encoding="utf-8"))
        cfg: dict[str, Any] = {}
        cfg_path = root / "tokenizer_config.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return cls(root, data, cfg)

    def _resolve_special_content(self, content: str) -> int:
        if content in self._added:
            return self._added[content]
        return int(self.vocab.get(content, self.unk_id))

    def _template_special_ids(self) -> tuple[list[int], list[int]]:
        node = self._post_processor
        if isinstance(node, dict) and node.get("type") == "Sequence":
            for proc in node.get("processors") or []:
                if isinstance(proc, dict) and proc.get("type") == "TemplateProcessing":
                    node = proc
                    break
        if not isinstance(node, dict) or node.get("type") != "TemplateProcessing":
            return [], []
        pre: list[int] = []
        post: list[int] = []
        seen_a = False
        for item in node.get("single") or []:
            if not isinstance(item, dict):
                continue
            seq = item.get("Sequence")
            if isinstance(seq, dict) and str(seq.get("id")) == "A":
                seen_a = True
                continue
            sp = item.get("SpecialToken")
            if not isinstance(sp, dict) or sp.get("id") is None:
                continue
            tid = self._resolve_special_content(str(sp["id"]))
            if not seen_a:
                pre.append(tid)
            else:
                post.append(tid)
        return pre, post

    def _bpe(self, word: str) -> list[str]:
        if not word:
            return []
        symbols = list(word)
        if len(symbols) == 1:
            return symbols
        while len(symbols) >= 2:
            ranks = [
                (self.merge_ranks.get((symbols[i], symbols[i + 1]), -1), i)
                for i in range(len(symbols) - 1)
            ]
            best_rank, best_i = min(
                ranks,
                key=lambda x: (x[0] if x[0] >= 0 else 1 << 30, x[1]),
            )
            if best_rank < 0:
                break
            i = best_i
            symbols[i : i + 2] = [symbols[i] + symbols[i + 1]]
        return symbols

    def _apply_pre_tokenizer_node(self, node: Any, pieces: list[str]) -> list[str]:
        if not isinstance(node, dict):
            return pieces
        kind = node.get("type")
        if kind == "Sequence":
            out = pieces
            for child in node.get("pretokenizers") or []:
                out = self._apply_pre_tokenizer_node(child, out)
            return out
        if kind == "Split":
            pattern = _split_pattern_from_node(node)
            out: list[str] = []
            for piece in pieces:
                out.extend(_split_with_pattern(piece, pattern))
            return out
        if kind == "ByteLevel":
            add_prefix_space = bool(node.get("add_prefix_space", False))
            return [
                _byte_level_pretokenize(piece, add_prefix_space=add_prefix_space)
                for piece in pieces
            ]
        raise RuntimeError(f"Unsupported pretokenizer type {kind!r} under {self.root}.")

    def _pretokenize_pieces(self, text: str) -> list[str]:
        if isinstance(self._pre_tokenizer, dict):
            return self._apply_pre_tokenizer_node(self._pre_tokenizer, [text])
        return re.findall(r"\S+|\s+", text)

    def _encode_piece(self, piece: str) -> list[int]:
        ids: list[int] = []
        for sub in self._bpe(piece):
            ids.append(int(self.vocab.get(sub, self.unk_id)))
        return ids

    def _encode_chunk(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in self._pretokenize_pieces(text):
            ids.extend(self._encode_piece(piece))
        return ids

    def _encode_text_with_specials(self, text: str) -> list[int]:
        if not self._added:
            return self._encode_chunk(text)
        specials = sorted(self._added.keys(), key=len, reverse=True)
        ids: list[int] = []
        i = 0
        while i < len(text):
            matched = False
            for sp in specials:
                if text.startswith(sp, i):
                    ids.append(self._added[sp])
                    i += len(sp)
                    matched = True
                    break
            if matched:
                continue
            next_i = len(text)
            for sp in specials:
                j = text.find(sp, i + 1)
                if j != -1:
                    next_i = min(next_i, j)
            ids.extend(self._encode_chunk(text[i:next_i]))
            i = next_i
        return ids

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = self._encode_text_with_specials(text)
        if add_special_tokens:
            pre, post = self._template_special_ids()
            ids = pre + ids + post
        return ids

    def encode_batch(
        self,
        texts: list[str],
        *,
        max_length: int,
        add_special_tokens: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        input_ids = np.zeros((len(texts), max_length), dtype=np.int32)
        attention_mask = np.zeros((len(texts), max_length), dtype=np.int32)
        for i, text in enumerate(texts):
            ids = self.encode(text, add_special_tokens=add_special_tokens)[:max_length]
            n = len(ids)
            input_ids[i, :n] = ids
            attention_mask[i, :n] = 1
        return input_ids, attention_mask


def render_qwen_chat_messages(messages: list[dict]) -> str:
    """Qwen2.5 text-only chat (HunyuanVideo system + user + generation prompt)."""
    blocks: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, list):
            text = "".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            text = str(content)
        blocks.append(f"<|im_start|>{role}\n{text}")
    return "\n".join(blocks) + "\n<|im_start|>assistant\n"


@lru_cache(maxsize=8)
def load_hf_tokenizer(tokenizer_dir: str) -> HFTokenizerJson:
    return HFTokenizerJson.from_directory(Path(tokenizer_dir))
