"""Z-Image 文本编码器 — PyTorch / CUDA 前向（形态 B：与 ``text_encoder_mlx`` 分离）。"""
from __future__ import annotations

from typing import Any


def zimage_prepare_torch_ids(ctx: Any, input_ids_np, attention_mask_np):
    import torch

    input_ids = torch.tensor(input_ids_np, dtype=torch.long, device=ctx._device)
    attention_mask = torch.tensor(attention_mask_np, dtype=torch.long, device=ctx._device)
    return input_ids, attention_mask


def zimage_text_encoder_forward_torch(encoder: Any, input_ids, attention_mask, num_valid: int) -> Any:
    import torch
    from transformers import AutoModel

    if encoder._model is None:
        encoder._model = AutoModel.from_pretrained(
            encoder.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(encoder.ctx._device)
        encoder._model.eval()

    with torch.no_grad():
        outputs = encoder._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    hs = outputs.hidden_states
    if hs is None:
        raise RuntimeError(
            "ZImageTextEncoder (torch): expected hidden_states; "
            "ensure transformers supports output_hidden_states for this checkpoint."
        )

    if encoder.hidden_state_layers is not None:
        layer_outputs = [hs[i] for i in encoder.hidden_state_layers]
        stacked = torch.stack(layer_outputs, dim=1)
        B, L, S, D = stacked.shape
        result = stacked.transpose(1, 2).reshape(B, S, L * D)
    else:
        result = hs[-2]

    result = result[:, :num_valid, :]
    return result.to(dtype=torch.bfloat16)


class ZImageTextEncoderCuda:
    """CUDA Z-Image text encoder (HF Qwen3 trunk + chat template)."""

    def __init__(
        self,
        ctx: Any,
        model_path: str,
        max_seq_len: int = 512,
        tokenizer_path: str = "",
        hidden_state_layers: tuple[int, ...] | None = None,
        enable_thinking: bool = False,
        **_kw: Any,
    ):
        self.ctx = ctx
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.max_seq_len = max_seq_len
        self.hidden_state_layers = hidden_state_layers
        self.enable_thinking = enable_thinking
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True,
            )
        return self._tokenizer

    def _tokenize_np(self, texts: list[str]) -> tuple[Any, Any, int]:
        tokenizer = self.tokenizer
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            chat_texts = []
            for text in texts:
                chat = [{"role": "user", "content": text}]
                chat_text = tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
                chat_texts.append(chat_text)
            tokens = tokenizer(
                chat_texts,
                padding="max_length",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="np",
            )
        else:
            tokens = tokenizer(
                texts,
                padding="max_length",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="np",
            )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        num_valid = int(attention_mask.sum())
        return input_ids, attention_mask, num_valid

    def encode(self, texts: list[str]) -> Any:
        input_ids, attention_mask, num_valid = self._tokenize_np(texts)
        tid, tam = zimage_prepare_torch_ids(self.ctx, input_ids, attention_mask)
        return zimage_text_encoder_forward_torch(self, tid, tam, num_valid)

    def release_weights(self) -> None:
        self._model = None
        self.ctx.clear_cache()


def zimage_encode_cuda(encoder: Any, texts: list[str]) -> Any:
    """Backward-compatible helper — prefer ``ZImageTextEncoderCuda.encode``."""
    return encoder.encode(texts)
