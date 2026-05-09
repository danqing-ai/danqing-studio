"""
Qwen2.5-VL Text Encoder for LongCat-Image。

使用 Qwen2Tokenizer + Qwen2.5-VL 模型获取文本嵌入。
权重路径: models/Base/longcat-image-fp16/text_encoder/
Tokenizer 路径: models/Base/longcat-image-fp16/text_processor/
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


class Qwen25VLEncoder:
    """Qwen2.5-VL 文本编码器。
    
    加载 transformers 模型，在 CPU 上运行，返回 MLX array。
    """
    def __init__(self, model_path: str | Path, device: str = "cpu"):
        from transformers import AutoModel, Qwen2Tokenizer
        import torch
        
        self.model_path = Path(model_path)
        self.device = device
        
        # Load from text_encoder subdirectory
        text_encoder_path = self.model_path / "text_encoder"
        text_processor_path = self.model_path / "text_processor"
        
        # Use Qwen2Tokenizer (not AutoTokenizer)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(str(text_processor_path))
        
        # Load model
        self.model = AutoModel.from_pretrained(
            str(text_encoder_path),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def encode(self, prompts: list[str], max_length: int = 512) -> mx.array:
        """编码文本提示，返回 [B, seq_len, 3584] 的 MLX array。"""
        import torch
        
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        
        # Forward to get hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use last layer hidden state
            hidden_states = outputs.hidden_states[-1]  # [B, seq_len, 3584]
        
        # Convert to numpy then MLX
        hidden_np = hidden_states.cpu().float().numpy()
        return mx.array(hidden_np, dtype=mx.float32)

    def __call__(self, prompts: list[str]) -> mx.array:
        return self.encode(prompts)
