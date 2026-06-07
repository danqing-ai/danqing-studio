#!/usr/bin/env python3
"""Normalize models_registry.json user-facing i18n labels.

- versions.*.name → bilingual {zh, en}; quant tier only (no ModelScope / HF in label)
- name / description → capability text only; download source is shown via badges
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "default_config" / "models_registry.json"

CJK_RE = re.compile(r"[\u4e00-\u9fff]")

DESCRIPTION_OVERRIDES: dict[str, dict[str, str]] = {
    "flux1-krea-dev": {
        "zh": "FLUX.1 Krea 合作版，增强真实感，12B 参数。",
        "en": "FLUX.1 Krea collaboration variant with enhanced realism; 12B parameters.",
    },
    "flux2-klein-4b": {
        "zh": "FLUX.2 Klein 4B 蒸馏版，4 步快速出图，质量优秀。",
        "en": "FLUX.2 Klein 4B distilled variant; fast 4-step generation with strong quality.",
    },
    "flux2-klein-base-4b": {
        "zh": "FLUX.2 Klein 4B 基础版，50 步高质量，支持训练。",
        "en": "FLUX.2 Klein 4B base model; high quality at 50 steps and trainable.",
    },
    "flux2-klein-9b": {
        "zh": "FLUX.2 Klein 9B 蒸馏版，8 步真实感出图。",
        "en": "FLUX.2 Klein 9B distilled; realistic 8-step generation.",
    },
    "flux2-klein-base-9b": {
        "zh": "FLUX.2 Klein 9B 基础版，50 步高质量。",
        "en": "FLUX.2 Klein 9B base model; high quality at 50 steps.",
    },
    "flux1-dev": {
        "zh": "FLUX.1 开发版，28 步高质量出图，12B 参数。",
        "en": "FLUX.1 dev variant; high quality at 28 steps, 12B parameters.",
    },
    "fibo-lite": {
        "zh": "Bria FIBO-Lite 轻量版，8B 参数，支持 JSON 结构化提示。",
        "en": "Bria FIBO-Lite lightweight 8B model with JSON-structured prompts.",
    },
    "fibo-edit": {
        "zh": "Bria FIBO-Edit 图像编辑模型，支持自然语言图像编辑。",
        "en": "Bria FIBO-Edit image editing with natural-language instructions.",
    },
    "fibo-edit-rmbg": {
        "zh": "Bria FIBO-Edit-RMBG 背景移除模型，输出透明 PNG。",
        "en": "Bria FIBO-Edit-RMBG background removal; outputs transparent PNG.",
    },
    "seedvr2-3b": {
        "zh": "SeedVR2 3B 超分辨率模型，用于图像放大与增强。",
        "en": "SeedVR2 3B upscaler for image enlargement and enhancement.",
    },
    "seedvr2-7b": {
        "zh": "SeedVR2 7B 超分辨率模型，更高质量的放大效果。",
        "en": "SeedVR2 7B upscaler for higher-quality enlargement.",
    },
    "depth-pro": {
        "zh": "Apple Depth Pro 深度估计模型，生成高质量深度图。",
        "en": "Apple Depth Pro depth estimation for high-quality depth maps.",
    },
    "flux-canny-controlnet": {
        "zh": "FLUX.1 Canny 边缘控制，让生成遵循边缘轮廓。",
        "en": "FLUX.1 Canny edge control for generation guided by edge maps.",
    },
    "flux-depth-controlnet": {
        "zh": "FLUX.1 Depth 深度控制，让生成遵循空间深度关系。",
        "en": "FLUX.1 Depth control for generation guided by depth structure.",
    },
    "flux-fill-controlnet": {
        "zh": "FLUX.1 Fill 局部重绘与图像修复/外扩。",
        "en": "FLUX.1 Fill for inpainting, retouch, and outpainting.",
    },
    "flux-redux": {
        "zh": "FLUX.1 Redux 图像变体与风格迁移，混合参考图与文本提示。",
        "en": "FLUX.1 Redux for image variation and style transfer from reference + prompt.",
    },
    "flux1-kontext": {
        "zh": "FLUX.1 Kontext 上下文编辑，支持多模态条件生成与编辑。",
        "en": "FLUX.1 Kontext for multimodal conditioned generation and editing.",
    },
    "qwen-image-edit": {
        "zh": "通义千问 Qwen Image Edit 指令编辑，VL 图文理解 + VAE 参考 latent。",
        "en": "Qwen Image Edit with VL understanding and VAE reference latents.",
    },
    "ace-step-xl-sft": {
        "zh": "ACE-Step-1.5 XL SFT，4B DiT 解码器，高质量音乐生成。",
        "en": "ACE-Step-1.5 XL SFT; 4B DiT decoder for high-quality music generation.",
    },
    "realism-lora-nsfw": {
        "zh": "适用于 FLUX.1-dev；真实感增强 LoRA，提升人物细节。",
        "en": "For FLUX.1-dev; realism enhancement LoRA for finer portrait detail.",
    },
    "miasweet-nsfw-lora": {
        "zh": "适用于 FLUX.1-dev；NSFW LoRA，触发词：miasweet。",
        "en": "For FLUX.1-dev; NSFW LoRA. Trigger word: miasweet.",
    },
    "bbw-style": {
        "zh": "适用于 FLUX.2 Klein 9B 的风格 LoRA。",
        "en": "Style LoRA for FLUX.2 Klein 9B.",
    },
    "lyza": {
        "zh": "Lyza 风格 LoRA。",
        "en": "Lyza style LoRA.",
    },
    "flux-uncensored-enhanceaiteam-v2-ms": {
        "zh": "enhanceaiteam Flux Uncensored V2 LoRA；适用于 FLUX.1-dev，成人向，请遵守法律与许可。",
        "en": "enhanceaiteam Flux Uncensored V2 LoRA for FLUX.1-dev; adult content—comply with law and licenses.",
    },
    "flux-uncensored-ryouko-v2-ms": {
        "zh": "Uncensored V2 社区变体 LoRA；适用于 FLUX.1-dev，成人向。",
        "en": "Uncensored V2 community variant LoRA for FLUX.1-dev; adult content.",
    },
    "flux-uncensored-lustlyai-v1-ms": {
        "zh": "Lustly.ai NSFW/Uncensored LoRA；适用于 FLUX.1-dev。",
        "en": "Lustly.ai NSFW/uncensored LoRA for FLUX.1-dev.",
    },
    "flux-uncensored-enhanceaiteam-v1-hf": {
        "zh": "enhanceaiteam Flux Uncensored V1 LoRA；适用于 FLUX.1-dev，成人向。",
        "en": "enhanceaiteam Flux Uncensored V1 LoRA for FLUX.1-dev; adult content.",
    },
    "flux-uncensored-enhanceaiteam-v2-hf": {
        "zh": "enhanceaiteam Uncensored V2 LoRA；与 V2 系列同源，适用于 FLUX.1-dev。",
        "en": "enhanceaiteam Uncensored V2 LoRA; same series as V2, for FLUX.1-dev.",
    },
    "flux-uncensored-the1ian-hf": {
        "zh": "the1ian Flux Uncensored LoRA；适用于 FLUX.1-dev，成人向。",
        "en": "the1ian Flux Uncensored LoRA for FLUX.1-dev; adult content.",
    },
    "flux1-dev-lora-antiblur": {
        "zh": "LiblibAI 去糊/清晰增强 LoRA，适用于 FLUX.1-dev。",
        "en": "LiblibAI anti-blur/sharpness LoRA for FLUX.1-dev.",
    },
    "flux1-dev-lora-ghibli": {
        "zh": "InstantX 吉卜力动画风格 LoRA，适用于 FLUX.1-dev。",
        "en": "InstantX Ghibli-style LoRA for FLUX.1-dev.",
    },
    "flux1-dev-lora-makoto-shinkai": {
        "zh": "InstantX 新海诚画风 LoRA，适用于 FLUX.1-dev。",
        "en": "InstantX Makoto Shinkai–style LoRA for FLUX.1-dev.",
    },
    "flux1-dev-lora-canny": {
        "zh": "官方 Canny 控制 LoRA，需配合边缘/Canny 条件流程。",
        "en": "Official Canny control LoRA; use with edge/Canny conditioning.",
    },
    "flux1-canny-dev-lora": {
        "zh": "官方 Canny 控制 LoRA，需配合边缘/Canny 条件流程。",
        "en": "Official Canny control LoRA; use with edge/Canny conditioning.",
    },
    "flux1-dev-lora-depth": {
        "zh": "官方 Depth 控制 LoRA，需配合深度图条件。",
        "en": "Official Depth control LoRA; requires depth conditioning.",
    },
    "flux2-klein-ac-style-lora": {
        "zh": "Klein AC 风格 LoRA；默认绑定 9B，若作者标明仅 4B 请改 base 或勿混用。",
        "en": "AC-style LoRA for FLUX.2 Klein; defaults to 9B—verify upstream if 4B-only.",
    },
    "flux2-klein-4b-spritesheet-lora": {
        "zh": "Klein 4B 精灵表/序列帧向 LoRA。",
        "en": "Spritesheet-oriented LoRA for Klein 4B.",
    },
    "flux2-klein-4b-zoom-lora": {
        "zh": "Klein 4B 变焦效果向 LoRA。",
        "en": "Zoom-effect LoRA for Klein 4B.",
    },
    "flux2-klein-4b-outpaint-lora": {
        "zh": "Klein 4B 画面外扩向 LoRA。",
        "en": "Outpainting LoRA for Klein 4B.",
    },
    "flux2-klein-4b-object-remove-lora": {
        "zh": "Klein 4B 物体移除向 LoRA。",
        "en": "Object-removal LoRA for Klein 4B.",
    },
    "starface-z-image-ms": {
        "zh": "明星脸/中国高颜值气质人像风格 LoRA，基于通义 Z-Image 训练。",
        "en": "Celebrity-style Chinese portrait LoRA trained on Tongyi Z-Image.",
    },
    "starface-qwen-image-2512-ms": {
        "zh": "明星脸/中国高颜值气质人像风格 LoRA，基于通义 Qwen Image 训练；引擎内合并仅 MLX。",
        "en": "Celebrity-style Chinese portrait LoRA for Qwen Image; in-engine merge is MLX-only.",
    },
    "hunyuan-video-1.5-480p-t2v": {
        "zh": "HunyuanVideo 1.5 文生视频，480p 文生变体。",
        "en": "HunyuanVideo 1.5 text-to-video; 480p T2V variant.",
    },
    "hunyuan-video-1.5-480p-i2v": {
        "zh": "HunyuanVideo 1.5 图生视频，480p 图生变体。",
        "en": "HunyuanVideo 1.5 image-to-video; 480p I2V variant.",
    },
    "hunyuan-video-1.5-i2v-step-distill": {
        "zh": "HunyuanVideo 1.5 图生视频蒸馏版，步数更少。",
        "en": "HunyuanVideo 1.5 distilled image-to-video; fewer steps.",
    },
    "hunyuan-video-1.5-1080p-sr": {
        "zh": "HunyuanVideo 1.5 超分修复，1080p 超分变体。",
        "en": "HunyuanVideo 1.5 super-resolution; 1080p SR variant.",
    },
    "ltx-2.3-distilled": {
        "zh": "Lightricks LTX-2.3 蒸馏版，含音视频 DiT，4 步快速生成。",
        "en": "Lightricks LTX-2.3 distilled A/V DiT; fast 4-step generation.",
    },
    "ltx-2.3-dev": {
        "zh": "Lightricks LTX-2.3 开发版，含音视频 DiT，支持 CFG，质量更高。",
        "en": "Lightricks LTX-2.3 dev A/V DiT; supports CFG for higher quality.",
    },
}

NAME_OVERRIDES: dict[str, dict[str, str]] = {
    "flux-uncensored-enhanceaiteam-v2-ms": {
        "zh": "Flux Uncensored V2",
        "en": "Flux Uncensored V2",
    },
    "flux-uncensored-ryouko-v2-ms": {
        "zh": "Flux Uncensored V2 Ryouko",
        "en": "Flux Uncensored V2 Ryouko",
    },
    "flux-uncensored-lustlyai-v1-ms": {
        "zh": "Lustly.ai Uncensored NSFW v1",
        "en": "Lustly.ai Uncensored NSFW v1",
    },
    "flux-uncensored-enhanceaiteam-v1-hf": {
        "zh": "Flux Uncensored V1",
        "en": "Flux Uncensored V1",
    },
    "flux-uncensored-enhanceaiteam-v2-hf": {
        "zh": "Flux Uncensored V2",
        "en": "Flux Uncensored V2",
    },
    "flux-uncensored-the1ian-hf": {
        "zh": "Flux Uncensored the1ian",
        "en": "Flux Uncensored the1ian",
    },
    "starface-z-image-ms": {
        "zh": "StarFace (Z-Image)",
        "en": "StarFace (Z-Image)",
    },
    "starface-qwen-image-2512-ms": {
        "zh": "StarFace (Qwen Image)",
        "en": "StarFace (Qwen Image)",
    },
}


def _bit_label(version_key: str, bits: int | None, old_plain: str) -> str | None:
    vk = version_key.lower()
    text = old_plain.lower()
    if vk == "mlx-bf16" or "bf16" in text:
        return "BF16"
    if bits == 6 or "6bit" in vk or "6-bit" in text or "6bit" in text:
        return "6-bit"
    if bits == 8 or vk.endswith("8bit") or vk == "mlx" or "8bit" in text or "int8" in text:
        return "8-bit"
    if bits == 4 or "4bit" in vk or vk == "mlx-q4" or "4bit" in text or "int4" in text:
        return "4-bit"
    return None


def _quant_variant(bit_label: str) -> dict[str, str]:
    if bit_label == "BF16":
        return {"zh": "BF16", "en": "BF16"}
    return {"zh": f"{bit_label} 量化版", "en": f"{bit_label} Quantized"}


def resolve_version_name(
    model_entry: dict[str, Any],
    version_key: str,
    version_entry: dict[str, Any],
) -> dict[str, str]:
    source_type = str(version_entry.get("source_type") or "full")
    quant = version_entry.get("quantization") or {}
    bits = quant.get("bits") if isinstance(quant, dict) else None
    old_name = version_entry.get("name")
    if isinstance(old_name, dict):
        old_plain = str(old_name.get("zh") or old_name.get("en") or "")
    else:
        old_plain = str(old_name or "")

    vk = version_key.lower()
    media = model_entry.get("media")

    if vk == "original" or old_plain == "原始权重":
        return {"zh": "原始权重", "en": "Original Weights"}

    if vk == "xl-sft":
        return {"zh": "XL SFT (4B)", "en": "XL SFT (4B)"}

    if media == "llm":
        if vk == "int4":
            return {"zh": "4-bit 量化版", "en": "4-bit Quantized"}
        if vk == "fp16":
            return {"zh": "FP16 完整版", "en": "FP16 Full"}

    if source_type == "derived":
        if bits == 8 or vk == "int8":
            return {"zh": "INT8 量化版", "en": "INT8 Quantized"}
        if bits == 4 or vk == "int4":
            return {"zh": "INT4 量化版", "en": "INT4 Quantized"}

    if source_type == "prequantized" or vk.startswith("mlx") or vk.startswith("community"):
        bit_label = _bit_label(vk, bits, old_plain)
        if bit_label:
            return _quant_variant(bit_label)

    if vk == "bf16" or old_plain.upper() == "BF16":
        return {"zh": "BF16", "en": "BF16"}

    if vk == "fp16" or old_plain in ("FP16", "FP16 完整版"):
        if old_plain == "FP16 完整版":
            return {"zh": "FP16 完整版", "en": "FP16 Full"}
        return {"zh": "FP16", "en": "FP16"}

    cleaned = clean_user_text(old_plain, lang="zh")
    if cleaned:
        return {"zh": cleaned, "en": clean_user_text(str(old_name.get("en") if isinstance(old_name, dict) else old_plain), lang="en") or cleaned}
    return {"zh": version_key, "en": version_key}


def clean_user_text(text: str, *, lang: str) -> str:
    text = str(text or "").strip()
    if not text:
        return text

    text = re.sub(
        r"[（(]\s*(?:魔搭|ModelScope|Hugging Face|HF)\s*(?:[）)]|$)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"[（(][^）)]*(?:魔搭|ModelScope|Hugging Face|\bHF\b)[^）)]*[）)]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?:魔搭|ModelScope|Hugging Face|\bHF\b)\s+[A-Za-z0-9._-]+/[A-Za-z0-9._/-]+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?:与魔搭|与 ModelScope)\s+[^；。,.]+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[，,]\s*[；;]", "；", text)
    text = re.sub(r"^[；;]\s*", "", text)
    text = re.sub(r"[；;]{2,}", "；", text)
    text = re.sub(r"[，,]{2,}", "，", text)
    text = text.strip(" ；;，,.")

    if lang == "en":
        text = re.sub(r"\bModelScope\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bHugging Face\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" ,.;")

    return text.strip()


def _ensure_sentence(text: str, *, lang: str) -> str:
    text = text.strip()
    if not text:
        return text
    if lang == "zh":
        if text[-1] in "。；":
            return text
        return f"{text}。"
    if text[-1] in ".;":
        return text
    return f"{text}."


def normalize_name(model_id: str, model_entry: dict[str, Any]) -> dict[str, str]:
    if model_id in NAME_OVERRIDES:
        return dict(NAME_OVERRIDES[model_id])

    name = model_entry.get("name") or {}
    if not isinstance(name, dict):
        raw = str(name or model_id)
        return {"zh": raw, "en": raw}

    zh = clean_user_text(str(name.get("zh") or model_id), lang="zh")
    en = clean_user_text(str(name.get("en") or zh), lang="en")
    return {"zh": zh or model_id, "en": en or zh or model_id}


def normalize_description(model_id: str, model_entry: dict[str, Any]) -> dict[str, str]:
    if model_id in DESCRIPTION_OVERRIDES:
        item = DESCRIPTION_OVERRIDES[model_id]
        return {
            "zh": _ensure_sentence(item["zh"], lang="zh"),
            "en": _ensure_sentence(item["en"], lang="en"),
        }

    desc = model_entry.get("description") or {}
    if not isinstance(desc, dict):
        raw = clean_user_text(str(desc or ""), lang="zh")
        return {"zh": _ensure_sentence(raw, lang="zh"), "en": _ensure_sentence(raw, lang="en")}

    zh = clean_user_text(str(desc.get("zh") or ""), lang="zh")
    en = clean_user_text(str(desc.get("en") or ""), lang="en")
    if CJK_RE.search(en):
        en = clean_user_text(str(desc.get("en") or ""), lang="en")
        if CJK_RE.search(en):
            en = zh
    if not en and zh:
        en = zh
    if not zh and en:
        zh = en

    return {
        "zh": _ensure_sentence(zh, lang="zh") if zh else "",
        "en": _ensure_sentence(en, lang="en") if en else "",
    }


def normalize_registry(data: dict[str, Any]) -> tuple[dict[str, Any], int]:
    out = json.loads(json.dumps(data))
    changes = 0
    models = out.get("models") or {}
    for model_id, model_entry in models.items():
        if not isinstance(model_entry, dict):
            continue

        new_name = normalize_name(model_id, model_entry)
        if model_entry.get("name") != new_name:
            model_entry["name"] = new_name
            changes += 1

        new_desc = normalize_description(model_id, model_entry)
        if model_entry.get("description") != new_desc:
            model_entry["description"] = new_desc
            changes += 1

        versions = model_entry.get("versions") or {}
        if not isinstance(versions, dict):
            continue
        for version_key, version_entry in versions.items():
            if not isinstance(version_entry, dict):
                continue
            new_vname = resolve_version_name(model_entry, version_key, version_entry)
            if version_entry.get("name") != new_vname:
                version_entry["name"] = new_vname
                changes += 1

    return out, changes


def main() -> int:
    path = REGISTRY_PATH
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    normalized, changes = normalize_registry(data)
    path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Normalized {path} ({changes} field updates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
