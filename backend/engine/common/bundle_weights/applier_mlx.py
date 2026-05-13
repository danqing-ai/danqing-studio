from __future__ import annotations

from typing import Any

from backend.engine.common.bundle_weights.loaded_weights import LoadedWeights


class WeightApplier:
    @staticmethod
    def set_weights(
        weights: LoadedWeights,
        models: dict[str, Any],
        components: dict | None = None,
    ) -> None:
        for name, model in models.items():
            component_weights = weights.components.get(name)
            if component_weights is not None:
                if components is not None:
                    component = components.get(name)
                    if component is not None and component.weight_subkey is not None:
                        component_weights = component_weights.get(
                            component.weight_subkey, component_weights
                        )
                model.update(component_weights, strict=False)
