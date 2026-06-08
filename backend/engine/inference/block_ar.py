"""BlockAutoregressiveInference — 分块自回归推理策略 (L2)。

外层 block-AR 循环管理 KV cache / EOS；内层 CFM 委托 ``FlowMatchingInference``。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from backend.engine.inference._protocols import AudioInferenceBundle, FlowMatchingSpec
from backend.engine.inference._runtime import raise_if_cancelled
from backend.engine.inference.flow_matching import FlowMatchingInference


class BlockAutoregressiveInference:
    """分块自回归推理 — 外层 block loop + 内层 flow-matching。"""

    def run(self, bundle: AudioInferenceBundle) -> dict[str, Any]:
        block = bundle.block_ar
        if block.setup_fn is None:
            raise RuntimeError("BlockAutoregressiveInference requires block_ar.setup_fn")

        if block.seed_fn is not None and bundle.seed:
            block.seed_fn(bundle.seed)

        setup_result = block.setup_fn() or {}
        num_blocks = block.num_blocks or setup_result.get("num_blocks", 0)
        if num_blocks <= 0:
            raise RuntimeError("BlockAutoregressiveInference requires block_ar.num_blocks > 0")

        inner_fm = FlowMatchingInference()
        sampled: Any = None

        for block_idx in range(num_blocks):
            raise_if_cancelled(bundle.cancel_token)

            if block.before_block_fn is not None:
                block.before_block_fn(block_idx)

            sub_bundle = replace(
                bundle,
                flow=FlowMatchingSpec(
                    timestep_schedule=bundle.flow.timestep_schedule,
                    init_noise_fn=bundle.flow.init_noise_fn,
                    euler_step_fn=bundle.flow.euler_step_fn,
                    cache_init_fn=bundle.flow.cache_init_fn,
                    before_step_fn=bundle.flow.before_step_fn,
                ),
                block_ar=type(block)(),
            )
            result = inner_fm.run(sub_bundle)
            sampled = result["latents"]

            if block.after_block_fn is not None:
                block.after_block_fn(block_idx, sampled)

            if block.eos_check_fn is not None and block.eos_check_fn(block_idx):
                break

        return {"latents": sampled}
