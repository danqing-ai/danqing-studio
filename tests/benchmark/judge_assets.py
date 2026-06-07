"""PickScore judge bundle — ModelScope download + local path resolution."""
from __future__ import annotations

import os
from pathlib import Path

from .registry_utils import resolve_benchmark_data_root

JUDGE_MODEL_ID_HF = "yuvalkirstain/PickScore_v1"
JUDGE_MODEL_ID_MS = "AI-ModelScope/PickScore_v1"
JUDGE_BUNDLE_REL = "models/Benchmark/pickscore-v1"


def judge_bundle_dir() -> Path:
    return (resolve_benchmark_data_root() / JUDGE_BUNDLE_REL).resolve()


def _bundle_ready(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def resolve_judge_model_path() -> str:
    """Local dir > env override > ModelScope/HF id (see ``judge_source()``)."""
    override = (os.environ.get("DANQING_BENCH_JUDGE_MODEL") or "").strip()
    if override:
        p = Path(override).expanduser().resolve()
        if not _bundle_ready(p):
            raise RuntimeError(f"judge_model_path_invalid:{p}")
        return str(p)

    local = judge_bundle_dir()
    if _bundle_ready(local):
        return str(local)

    source = judge_source()
    if source == "modelscope":
        return str(download_pickscore_modelscope())
    return JUDGE_MODEL_ID_HF


def judge_source() -> str:
    raw = (os.environ.get("DANQING_BENCH_JUDGE_SOURCE") or "modelscope").strip().lower()
    if raw in {"modelscope", "ms", "魔塔"}:
        return "modelscope"
    if raw in {"huggingface", "hf"}:
        return "huggingface"
    if raw == "local":
        return "local"
    raise RuntimeError(f"judge_unknown_source:{raw!r}")


def download_pickscore_modelscope(*, force: bool = False) -> Path:
    """Download PickScore to ``{workspace}/models/Benchmark/pickscore-v1`` via ModelScope."""
    dest = judge_bundle_dir()
    if not force and _bundle_ready(dest):
        return dest

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "judge_missing_modelscope: pip install modelscope (or use project .venv)"
        ) from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    if force and dest.exists():
        import shutil

        shutil.rmtree(dest)
    print(f"[judge] ModelScope download {JUDGE_MODEL_ID_MS} -> {dest}")
    snapshot_download(JUDGE_MODEL_ID_MS, local_dir=str(dest))
    if not _bundle_ready(dest):
        raise RuntimeError(f"judge_modelscope_incomplete:{dest}")
    print(f"[judge] ready at {dest}")
    return dest
