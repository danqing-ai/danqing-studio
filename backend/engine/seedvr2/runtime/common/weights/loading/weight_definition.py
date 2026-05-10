from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, TypeAlias

import mlx.core as mx

from backend.engine.seedvr2.runtime.common.weights.mapping.weight_mapping import WeightTarget

if TYPE_CHECKING:
    from backend.engine.seedvr2.runtime.seedvr2_pkg.weights.seedvr2_weight_definition import SeedVR2WeightDefinition

    WeightDefinitionType: TypeAlias = type[SeedVR2WeightDefinition]


@dataclass
class ComponentDefinition:
    name: str
    hf_subdir: str
    mapping_getter: Callable[[], List[WeightTarget]] | None = None
    model_attr: str | None = None
    num_blocks: int | None = None
    num_layers: int | None = None
    loading_mode: str = "mlx_native"
    precision: mx.Dtype | None = None
    skip_quantization: bool = False
    bulk_transform: Callable[[mx.array], mx.array] | None = None
    weight_subkey: str | None = None
    download_url: str | None = None
    weight_prefix_filters: List[str] | None = None
    weight_files: List[str] | None = None


@dataclass
class TokenizerDefinition:
    name: str
    hf_subdir: str
    tokenizer_class: str = "AutoTokenizer"
    fallback_subdirs: List[str] | None = None
    download_patterns: List[str] | None = None
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
