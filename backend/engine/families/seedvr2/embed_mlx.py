from __future__ import annotations

"""SeedVR2 超分固定正嵌入（无运行时 tokenizer）。

默认使用本包 ``data/pos_emb.safetensors``（与 `mflux` 仓库
``seedvr2_text_encoder/embeddings/pos_emb.safetensors`` 同源，形状 ``(58, 5120)``）。
若权重目录下放置同名文件，则优先从 bundle 加载。"""

from pathlib import Path

import mlx.core as mx
from backend.engine.common.mlx_runtime_fallback import load_weights_dict


def _package_pos_emb_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "pos_emb.safetensors"


def resolve_pos_emb_path(bundle_path: str | Path | None) -> Path:
    """解析 ``pos_emb.safetensors``：bundle 内可选覆盖，否则使用包内默认。"""
    candidates: list[Path] = []
    if bundle_path is not None:
        b = Path(bundle_path)
        candidates.extend(
            [
                b / "pos_emb.safetensors",
                b / "data" / "pos_emb.safetensors",
            ]
        )
    candidates.append(_package_pos_emb_path())
    for p in candidates:
        if p.is_file():
            return p
    tried = ", ".join(str(c) for c in candidates)
    raise RuntimeError(
        "SeedVR2 requires pos_emb.safetensors (fixed positive text embeddings). "
        f"None of the following paths exist: {tried}. "
        "Place a copy next to the weight bundle or under bundle/data/, "
        "or reinstall DanQing Studio so `backend/engine/families/seedvr2/data/` is present."
    )


class SeedVR2PositiveEmbeddings:
    """加载 ``pos_emb.safetensors`` 中的常量 ``txt`` 侧嵌入。"""

    @staticmethod
    def load(batch_size: int = 1, *, bundle_path: str | Path | None = None) -> mx.array:
        emb = load_weights_dict(None, str(resolve_pos_emb_path(bundle_path)))["embedding"]
        if emb.ndim == 2:
            emb = emb[None, ...]
        if batch_size > 1:
            emb = mx.repeat(emb, batch_size, axis=0)
        return emb
