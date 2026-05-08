"""
Qwen2.5-VL Text Encoder for LongCat-Image。

使用 transformers 库在 PyTorch 上运行，输出转换为 MLX array。
权重路径: models/Base/longcat-image-fp16/text_encoder/
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx


class Qwen25VLEncoder:
    """Qwen2.5-VL 文本编码器。
    
    加载 transformers 模型，在 CPU/MPS 上运行，返回 MLX array。
    """
    def __init__(self, model_path: str | Path, device: str = "cpu"):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        self.model_path = Path(model_path)
        self.device = device
        
        # 加载模型和处理器
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(self.model_path),
            torch_dtype="auto",
            device_map="cpu" if device == "cpu" else "auto",
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(str(self.model_path))
        self.model.eval()

    def encode(self, prompts: list[str], max_length: int = 512) -> mx.array:
        """编码文本提示，返回 [B, seq_len, 3584] 的 MLX array。"""
        import torch
        
        # 构建对话格式（Qwen2.5-VL 需要特定格式）
        conversations = []
        for prompt in prompts:
            conversations.append([
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ])
        
        # 使用处理器处理文本
        texts = [
            self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]
        
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        
        # 前向传播获取 hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # 使用最后一层 hidden state
            hidden_states = outputs.hidden_states[-1]  # [B, seq_len, 3584]
        
        # 转换为 numpy 再转 MLX
        hidden_np = hidden_states.cpu().float().numpy()
        return mx.array(hidden_np, dtype=mx.float32)

    def __call__(self, prompts: list[str]) -> mx.array:
        return self.encode(prompts)
