#!/usr/bin/env python3
"""One-shot: remove distribution.dependencies; fold assets into bundle_repos."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "default_config/models_registry.json"

WAN_MS_ENCODERS = {
    "repo_id": "Wan-AI/Wan2.2-T2V-A14B",
    "source": "modelscope",
    "size": "12GB",
    "name": {"zh": "Wan UMT5 + VAE", "en": "Wan UMT5 + VAE"},
    "allow_patterns": [
        "Wan2.1_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "google/**",
        "configuration.json",
    ],
}

HUNYUAN_DISTILL_ENCODER_REPOS = [
    {
        "repo_id": "Tencent-Hunyuan/HunyuanVideo-1.5",
        "source": "modelscope",
        "size": "12GB",
        "name": "HunyuanVideo-1.5 encoders",
        "allow_patterns": [
            "transformer/480p_t2v/config.json",
            "vae/**",
        ],
    },
    {
        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "local_path": "models/Text/qwen2.5-vl-7b-instruct",
        "source": "modelscope",
        "size": "16GB",
        "name": "Qwen2.5-VL-7B-Instruct",
    },
    {
        "repo_id": "google/byt5-small",
        "local_path": "models/Text/byt5-small",
        "source": "modelscope",
        "size": "1GB",
        "name": "byt5-small",
    },
]


def _primary_repo(ver: dict, *, local_path: str, name: str | dict) -> dict:
    entry: dict = {
        "repo_id": ver["repo_id"],
        "local_path": local_path,
        "name": name,
        "source": ver.get("source") or "modelscope",
    }
    if ver.get("size"):
        entry["size"] = ver["size"]
    if ver.get("allow_patterns"):
        entry["allow_patterns"] = ver["allow_patterns"]
    return entry


def _with_wan_encoders(ver: dict, *, name: str | dict) -> list[dict]:
    lp = ver["local_path"]
    enc = deepcopy(WAN_MS_ENCODERS)
    enc["local_path"] = lp
    return [_primary_repo(ver, local_path=lp, name=name), enc]


def main() -> None:
    data = json.loads(REG.read_text(encoding="utf-8"))
    models = data["models"]

    # --- Z-Image controlnet (adapter only; base model installed separately) ---
    zcn = models["z-image-turbo-fun-controlnet-union"]["distribution"]["versions"]["8steps"]
    zcn.pop("bundle_repos", None)
    zcn["size"] = "8GB"

    # --- Hunyuan distill ---
    hy = models["hunyuan-video-1.5-t2v-distill"]
    hy["catalog"]["description"] = {
        "zh": "HunyuanVideo 1.5 文生视频 LightX2V 4 步真蒸馏（单文件 DiT，480p）；完整自包含 bundle（DiT + VAE + 文本编码器）。",
        "en": "HunyuanVideo 1.5 T2V LightX2V 4-step distill (480p); self-contained bundle (DiT + VAE + text encoders).",
    }
    hy_bf16 = hy["distribution"]["versions"]["bf16"]
    hy_bf16["size"] = "46GB"
    hy_lp = hy_bf16["local_path"]
    hy_bf16["bundle_repos"] = [
        _primary_repo(
            hy_bf16,
            local_path=hy_lp,
            name={"zh": "Hy1.5 Distill DiT", "en": "Hy1.5 Distill DiT"},
        ),
    ]
    for enc in HUNYUAN_DISTILL_ENCODER_REPOS:
        row = deepcopy(enc)
        if "local_path" not in row:
            row["local_path"] = hy_lp
        hy_bf16["bundle_repos"].append(row)

    # --- Wan distill / turbo ---
    wan_updates: dict[str, dict[str, tuple[str, dict]]] = {
        "wan-2.2-t2v-14b-distill": {
            "fp8": (
                "46GB",
                {"zh": "Wan T2V Distill DiT", "en": "Wan T2V Distill DiT"},
            ),
        },
        "wan-2.2-i2v-14b-distill": {
            "bf16": (
                "126GB",
                {"zh": "Wan I2V Distill DiT", "en": "Wan I2V Distill DiT"},
            ),
        },
        "wan-2.2-i2v-14b-turbo": {
            "fp16": (
                "42GB",
                {"zh": "Turbo I2V DiT", "en": "Turbo I2V DiT"},
            ),
            "int8": (
                "28GB",
                {"zh": "Turbo I2V DiT INT8", "en": "Turbo I2V DiT INT8"},
            ),
        },
        "wan-2.2-t2v-1.3b-turbo": {
            "fp16": (
                "15GB",
                {"zh": "Turbo T2V 1.3B DiT", "en": "Turbo T2V 1.3B DiT"},
            ),
        },
        "wan-2.2-t2v-14b-turbo-720p": {
            "fp16": (
                "40GB",
                {"zh": "Turbo T2V 720p DiT", "en": "Turbo T2V 720p DiT"},
            ),
        },
        "wan-2.2-t2v-14b-turbo-480p": {
            "fp16": (
                "40GB",
                {"zh": "Turbo T2V 480p DiT", "en": "Turbo T2V 480p DiT"},
            ),
        },
    }

    for mid, versions in wan_updates.items():
        dist = models[mid]["distribution"]
        for vk, (size, name) in versions.items():
            ver = dist["versions"][vk]
            ver["size"] = size
            ver["bundle_repos"] = _with_wan_encoders(ver, name=name)

    # --- Strip all distribution.dependencies ---
    for mid, model in models.items():
        dist = model.get("distribution")
        if not isinstance(dist, dict):
            continue
        if dist.pop("dependencies", None) is not None:
            print(f"removed dependencies from {mid}")

    REG.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("registry migrated")


if __name__ == "__main__":
    main()
