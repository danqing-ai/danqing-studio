"""
文本编码器 — T5 / CLIP 编码器。

所有扩散模型的文本编码共享此实现。
参考 mflux 项目的 T5Encoder 实现和 mlx-video 的 T5 加载。
"""
from __future__ import annotations

from typing import Any, Optional


class T5Encoder:
    """T5-XXL 文本编码器。

    用于: Flux1 / Flux2 / LTX / Wan / CogVideoX 全系列。
    文本 → [B, seq_len, text_dim] 嵌入。
    """

    def __init__(self, ctx: Any, model_path: str,
                 max_seq_len: int = 512, text_dim: int = 4096):
        self.ctx = ctx
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self.text_dim = text_dim
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import T5Tokenizer
            self._tokenizer = T5Tokenizer.from_pretrained(
                self.model_path, legacy=False,
            )
        return self._tokenizer

    def encode(self, texts: list[str]) -> Any:
        """编码文本列表。

        Args:
            texts: 文本列表

        Returns:
            prompt_embeds: [B, seq_len, text_dim]
        """
        tokenizer = self.tokenizer
        ctx = self.ctx

        tokens = tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="np",
        )

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # 用 ctx 创建张量
        if ctx.backend == "mlx":
            import mlx.core as mx
            input_ids = mx.array(input_ids, dtype=mx.int32)
            attention_mask = mx.array(attention_mask, dtype=mx.float32)
        else:
            import torch
            input_ids = torch.tensor(input_ids, dtype=torch.int32, device=ctx._device)
            attention_mask = torch.tensor(attention_mask, dtype=torch.float32, device=ctx._device)

        # 前向通过 T5
        embeds = self._forward(input_ids, attention_mask)
        return embeds

    def _forward(self, input_ids, attention_mask):
        """占位：T5 前向。具体实现由各后端完成。

        MLX T5 → mlx_lm.models.t5.T5Model
        CUDA T5 → transformers.T5EncoderModel
        """
        if self.ctx.backend == "mlx":
            return self._forward_mlx(input_ids, attention_mask)
        else:
            return self._forward_torch(input_ids, attention_mask)

    def _forward_mlx(self, input_ids, attention_mask):
        import mlx.core as mx
        try:
            from mlx_lm.models.t5 import T5Model, T5Config
        except ImportError:
            raise ImportError("mlx_lm not installed. Install with: pip install mlx-lm")
        if self._model is None:
            config = T5Config.from_pretrained(self.model_path)
            self._model = T5Model(config)
            self._model.load_weights(str(self.model_path))
            mx.eval(self._model.parameters())
        return self._model(input_ids, attention_mask)

    def _forward_torch(self, input_ids, attention_mask):
        import torch
        from transformers import T5EncoderModel

        if self._model is None:
            self._model = T5EncoderModel.from_pretrained(
                self.model_path, torch_dtype=torch.float32
            ).to(self.ctx._device)
            self._model.eval()

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return outputs.last_hidden_state


class CLIPEncoder:
    """CLIP 文本/图像编码器。

    用于: Flux1 系列的双编码器（T5 + CLIP）。
    """

    def __init__(self, ctx: Any, model_path: str,
                 max_seq_len: int = 77, embed_dim: int = 768):
        self.ctx = ctx
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import CLIPTokenizer
            self._tokenizer = CLIPTokenizer.from_pretrained(self.model_path)
        return self._tokenizer

    def encode(self, texts: list[str]) -> tuple[Any, Any]:
        """编码文本 → (pooled_output, hidden_states)。

        Returns:
            pooled: [B, embed_dim] — 用于 timestep embedding 注入
            hidden: [B, seq_len, embed_dim] — 用于交叉注意力
        """
        tokenizer = self.tokenizer
        ctx = self.ctx

        tokens = tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="np",
        )

        input_ids = tokens["input_ids"]

        if ctx.backend == "mlx":
            import mlx.core as mx
            input_ids = mx.array(input_ids, dtype=mx.int32)
        else:
            import torch
            input_ids = torch.tensor(input_ids, dtype=torch.int32, device=ctx._device)

        return self._forward(input_ids)

    def _forward(self, input_ids):
        if self.ctx.backend == "mlx":
            return self._forward_mlx(input_ids)
        else:
            return self._forward_torch(input_ids)

    def _forward_mlx(self, input_ids):
        # MLX CLIP 文本编码器
        import mlx.core as mx
        from mlx_lm.models.clip import CLIPTextModel, CLIPTextConfig

        if self._model is None:
            config = CLIPTextConfig.from_pretrained(self.model_path)
            self._model = CLIPTextModel(config)
            self._model.load_weights(str(self.model_path))
            mx.eval(self._model.parameters())

        pooled, hidden = self._model(input_ids)
        return pooled, hidden

    def _forward_torch(self, input_ids):
        import torch
        from transformers import CLIPTextModel

        if self._model is None:
            self._model = CLIPTextModel.from_pretrained(
                self.model_path, torch_dtype=torch.float32
            ).to(self.ctx._device)
            self._model.eval()

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids)
        return outputs.pooler_output, outputs.last_hidden_state


class Qwen3Encoder:
    """Qwen3 文本编码器 — 直接加载 safetensors 权重。

    用于: Z-Image 系列。
    model_path: text_encoder 权重目录
    tokenizer_path: tokenizer 目录 (可与 model_path 不同)
    """

    def __init__(self, ctx: Any, model_path: str,
                 max_seq_len: int = 512, embed_dim: int = 2560,
                 tokenizer_path: str = ""):
        self.ctx = ctx
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, trust_remote_code=True,
            )
        return self._tokenizer

    def encode(self, texts: list[str]) -> Any:
        tokenizer = self.tokenizer
        ctx = self.ctx

        tokens = tokenizer(
            texts, padding="max_length", max_length=self.max_seq_len,
            truncation=True, return_tensors="np",
        )
        input_ids = tokens["input_ids"]

        if ctx.backend == "mlx":
            import mlx.core as mx
            input_ids = mx.array(input_ids, dtype=mx.int32)
        else:
            import torch
            input_ids = torch.tensor(input_ids, dtype=torch.int32, device=ctx._device)

        return self._forward(input_ids)

    def _forward(self, input_ids):
        if self.ctx.backend == "mlx":
            return self._forward_mlx(input_ids)
        else:
            return self._forward_torch(input_ids)

    def _forward_mlx(self, input_ids):
        import mlx.core as mx
        # 创建 Qwen3 模型并加载权重（简化：使用 MLX 原生模块）
        if self._model is None:
            self._model = self._build_qwen3_model()
        return self._model(input_ids)

    def _build_qwen3_model(self):
        """从 text_encoder/ 目录的 safetensors 构建 Qwen3 模型。"""
        import mlx.core as mx
        import mlx.nn as nn
        from pathlib import Path

        model_dir = Path(self.model_path)
        # 收集所有权重
        weights = {}
        for sf in sorted(model_dir.glob("*.safetensors")):
            weights.update(dict(mx.load(str(sf))))

        # 从 config.json 读取配置
        import json
        config_path = model_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}

        hidden_size = config.get("hidden_size", self.embed_dim)
        num_layers = config.get("num_hidden_layers", 32)
        num_heads = config.get("num_attention_heads", 32)
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        vocab_size = config.get("vocab_size", 151936)
        intermediate_size = config.get("intermediate_size", hidden_size * 8 // 3)

        # 构建简易 Qwen3 模型（Embedding + Transformer layers + RMSNorm）
        class Qwen3Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
                self.layers = [
                    _Qwen3DecoderLayer(hidden_size, num_heads, num_kv_heads, intermediate_size)
                    for _ in range(num_layers)
                ]
                self.norm = nn.RMSNorm(hidden_size)

            def __call__(self, input_ids):
                h = self.embed_tokens(input_ids)
                for layer in self.layers:
                    h = layer(h)
                h = self.norm(h)
                return h

        class _Qwen3DecoderLayer(nn.Module):
            def __init__(self, dim, n_heads, n_kv, ffn_dim):
                super().__init__()
                head_dim = dim // n_heads
                self.self_attn = nn.MultiHeadAttention(dim, n_heads)
                self.input_layernorm = nn.RMSNorm(dim)
                self.post_attention_layernorm = nn.RMSNorm(dim)
                self.mlp = nn.Sequential(
                    nn.Linear(dim, ffn_dim),
                    nn.SiLU(),
                    nn.Linear(ffn_dim, dim),
                )

            def __call__(self, x):
                r = self.self_attn(self.input_layernorm(x), self.input_layernorm(x), self.input_layernorm(x))
                x = x + r
                x = x + self.mlp(self.post_attention_layernorm(x))
                return x

        model = Qwen3Model()
        # 键名映射: diffusers text_encoder key → MLX nn.Module key
        # model.embed_tokens.weight ← model.embed_tokens.weight
        # model.layers.0.input_layernorm.weight ← model.layers.0.input_layernorm.weight
        remapped = {}
        for key, tensor in weights.items():
            new_key = key.replace("model.", "")
            remapped[new_key] = tensor

        model.load_weights(list(remapped.items()), strict=False)
        mx.eval(model.parameters())
        return model

    def _forward_torch(self, input_ids):
        import torch
        from transformers import AutoModel
        if self._model is None:
            self._model = AutoModel.from_pretrained(
                self.model_path, torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to(self.ctx._device)
            self._model.eval()
        with torch.no_grad():
            return self._model(input_ids=input_ids).last_hidden_state
