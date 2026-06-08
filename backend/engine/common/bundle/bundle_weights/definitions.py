from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from backend.engine.common.bundle.weight_mapping import WeightTarget


@dataclass
class ComponentDefinition:
    name: str
    hf_subdir: str
    mapping_getter: Callable[[], list[WeightTarget]] | None = None
    model_attr: str | None = None
    num_blocks: int | None = None
    num_layers: int | None = None
    loading_mode: str = "mlx_native"
    precision: Any | None = None
    skip_quantization: bool = False
    bulk_transform: Callable[[Any], Any] | None = None
    weight_subkey: str | None = None
    download_url: str | None = None
    weight_prefix_filters: list[str] | None = None
    weight_files: list[str] | None = None


@dataclass
class TokenizerDefinition:
    name: str
    hf_subdir: str
    tokenizer_class: str = "AutoTokenizer"
    fallback_subdirs: list[str] | None = None
    download_patterns: list[str] | None = None
    encoder_class: type | None = None
    max_length: int = 512
    padding: str = "max_length"
    template: str | None = None
    use_chat_template: bool = False
    chat_template_kwargs: dict | None = field(default_factory=dict)
    add_special_tokens: bool = True
    processor_class: type | None = None
    image_token: str = "<|image_pad|>"
    chat_template: str | None = None
