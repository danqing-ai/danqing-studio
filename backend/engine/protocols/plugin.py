"""Family plugin aggregate — v3 family-facing surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from backend.engine.protocols.components import Backbone, EncodeResult, TextEncoder, VAE

LatentLayout = Literal["nchw", "packed_seq", "qwen_grid", "video_5d", "audio_1d"]
CfgMode = Literal["fused", "batched", "dual", "none"]
ParadigmKind = Literal["diffusion", "flow_matching", "block_ar", "two_stage", "job"]


@dataclass(frozen=True)
class FamilySpec:
    """Runtime semantics — sourced from catalog ``families`` block (v3 registry)."""

    family_id: str
    media: Literal["image", "video", "audio"]
    paradigm: ParadigmKind = "diffusion"
    latent_layout: LatentLayout = "nchw"
    latent_channels: int = 16
    vae_scale: int = 8
    cfg_mode: CfgMode = "dual"
    supports_guidance: bool = True
    default_scheduler: str = "flow_match_euler"
    step_kwargs_profile: str | None = None
    hooks: frozenset[str] = frozenset()
    backends: tuple[str, ...] = ("mlx",)


ConditioningFn = Callable[..., Any]
ExtraCondFn = Callable[..., dict[str, Any]]
ParadigmSelectorFn = Callable[[FamilySpec], ParadigmKind]


@dataclass
class FamilyPlugin:
    """Composable family entry — sessions and inference depend on this, not family strings."""

    family_id: str
    spec: FamilySpec
    backbone: Backbone
    vae: VAE | None = None
    text_encoder: TextEncoder | None = None
    secondary_encoders: dict[str, TextEncoder] = field(default_factory=dict)
    encode_conditioning: ConditioningFn | None = None
    build_extra_cond: ExtraCondFn | None = None

    def select_paradigm(self) -> ParadigmKind:
        return self.spec.paradigm
