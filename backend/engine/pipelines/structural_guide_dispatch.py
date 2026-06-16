"""Family dispatch for ``ImageGenerationRequest.structural_guide`` hooks."""
from __future__ import annotations

from typing import Any


def augment_request_for_structural_guide(request: Any, ctx: Any) -> Any:
    guide = getattr(request, "structural_guide", None)
    if guide is None:
        return request
    model_id = (getattr(request, "model", None) or "").split(":", 1)[0].strip()
    if model_id.startswith("flux"):
        from backend.engine.families.flux1.structural import augment_request_for_structural_guide as flux_augment

        return flux_augment(request, ctx)
    if model_id in ("z-image", "z-image-turbo") or model_id.startswith("z-image"):
        from backend.engine.families.z_image.structural import augment_request_for_structural_guide as z_augment

        return z_augment(request, ctx)
    raise RuntimeError(
        f"structural_guide is not supported for model={model_id!r}; "
        "use flux1-* or z-image / z-image-turbo with a compatible controlnet bundle"
    )


def attach_structural_conditioning(
    pipeline: Any,
    *,
    request: Any,
    family: str,
    model: Any,
    entry: Any,
    version_key: str | None,
    extra_cond: dict[str, Any],
    width: int,
    height: int,
    ctx_exec: Any,
    on_log: Any | None,
) -> tuple[dict[str, Any], Any | None]:
    if family == "flux1":
        from backend.engine.families.flux1.structural import attach_structural_conditioning as flux_attach

        return flux_attach(
            pipeline,
            request=request,
            family=family,
            model=model,
            entry=entry,
            version_key=version_key,
            extra_cond=extra_cond,
            width=width,
            height=height,
            ctx_exec=ctx_exec,
            on_log=on_log,
        )
    if family == "z_image":
        from backend.engine.families.z_image.structural import attach_structural_conditioning as z_attach

        return z_attach(
            pipeline,
            request=request,
            family=family,
            model=model,
            entry=entry,
            version_key=version_key,
            extra_cond=extra_cond,
            width=width,
            height=height,
            ctx_exec=ctx_exec,
            on_log=on_log,
        )
    raise RuntimeError(
        f"structural_guide is not supported for family={family!r} "
        f"(ModelConfig.supports_structural_guide=false or unknown family)"
    )
