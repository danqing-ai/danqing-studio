#!/usr/bin/env python3
"""根据上游仓库公开元数据推断 ``commercial_use_allowed``（维护用脚本）。

从 Hugging Face ``/api/models/{repo_id}`` 与魔搭 ``HubApi.get_model`` 读取 license / tags，
再映射到 ``true`` / ``false`` / ``null``（无法可靠判断时保持或写入 null）。

限制（必读）：
- Hub 上的 ``license`` 字段可能缺失、笼统（如 ``other``）或与真实法律文件不一致。
- LoRA/适配器与 BFL NC 底模组合使用时的可商用性，**不能**仅由 LoRA 仓库 license 代表。
- 本脚本输出**非法律意见**；发布前仍应对照各模型官方许可文本。

用法：
  .venv/bin/python scripts/infer_registry_commercial_use.py --dry-run
  .venv/bin/python scripts/infer_registry_commercial_use.py --write
  HF_ENDPOINT=https://hf-mirror.com .venv/bin/python scripts/infer_registry_commercial_use.py --write

``--write`` 默认只更新「本次能给出 true/false 推断」的条目；已有明确布尔且加 ``--force`` 时才覆盖。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "config" / "models_registry.json"

# 出现在 license 字符串或 license: 标签中则倾向「不可商用」或需单独签署协议
_NON_COMMERCIAL_PATTERNS = (
    r"non[\s-]?commercial",
    r"\bnc\b",
    r"cc[\s-]?by[\s-]?nc",
    r"research[\s-]?only",
    r"personal[\s-]?use",
    r"non[\s-]?profit",
    r"academic[\s-]?only",
    r"flux[\s._-]?1[\s._-]?dev",  # BFL FLUX.1 Dev 非商用许可常见关键字
    r"black[\s-]?forest[\s-]?labs[\s-]?non[\s-]?commercial",
)

# SPDX 或 Hub 常见「偏宽松」许可（仍可能受模型行为条款约束；仅作粗分类）
_PERMISSIVE_LICENSES = frozenset(
    {
        "apache-2.0",
        "apache2",
        "mit",
        "bsd",
        "bsd-2-clause",
        "bsd-3-clause",
        "isc",
        "cc0-1.0",
        "cc0",
        "unlicense",
        "wtfpl",
    }
)


def _norm_license_token(s: str) -> str:
    return re.sub(r"[\s_]+", "-", s.strip().lower())


def _collect_hf_licenses(payload: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for t in payload.get("tags") or []:
        if not isinstance(t, str):
            continue
        tl = t.lower()
        if tl.startswith("license:"):
            out.append(t.split(":", 1)[1].strip())
    cd = payload.get("cardData") or {}
    lic = cd.get("license")
    if isinstance(lic, str):
        out.append(lic.strip())
    elif isinstance(lic, list):
        for x in lic:
            if isinstance(x, str):
                out.append(x.strip())
    return out


def _infer_from_license_strings(licenses: List[str]) -> Optional[bool]:
    if not licenses:
        return None
    blob = " | ".join(licenses)
    for pat in _NON_COMMERCIAL_PATTERNS:
        if re.search(pat, blob, re.I):
            return False

    tokens: List[str] = []
    for L in licenses:
        for part in re.split(r"[,;/|]+", L):
            t = _norm_license_token(part)
            if t and t not in tokens:
                tokens.append(t)
    meaningful = [t for t in tokens if t not in ("other", "unknown", "none")]
    if not meaningful:
        return None

    def _is_permissive(t: str) -> bool:
        if t in _PERMISSIVE_LICENSES:
            return True
        if t.startswith("apache-") or t.startswith("bsd-"):
            return True
        return False

    if all(_is_permissive(t) for t in meaningful):
        return True
    return None


def _fetch_hf_model(repo_id: str, endpoint: str, timeout: float) -> Optional[Dict[str, Any]]:
    base = endpoint.rstrip("/")
    url = f"{base}/api/models/{repo_id}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "DanQing-Studio/infer_registry_commercial_use"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None


def _fetch_ms_license(model_id: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        from modelscope.hub.api import HubApi
    except ImportError:
        return None, "modelscope package not installed"
    try:
        m = HubApi().get_model(model_id, revision="master")
    except Exception as e:
        return None, str(e)
    if not isinstance(m, dict):
        return None, "unexpected get_model response"
    lic = m.get("License") or m.get("license")
    if lic is None:
        return None, None
    return str(lic).strip(), None


def _default_version_key(versions: Dict[str, Any]) -> Optional[str]:
    if not isinstance(versions, dict) or not versions:
        return None
    for k, v in versions.items():
        if isinstance(v, dict) and v.get("default") is True:
            return k
    return next(iter(versions.keys()), None)


def _repo_from_record(rec: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Returns (repo_id, version_key_used)."""
    ver = rec.get("versions")
    if isinstance(ver, dict):
        vk = _default_version_key(ver)
        if vk:
            block = ver.get(vk)
            if isinstance(block, dict):
                rid = block.get("repo_id")
                if isinstance(rid, str) and rid.strip():
                    return rid.strip(), vk
    return None, None


def _policy_false_commercial(repo_id: str) -> bool:
    """BFL FLUX.1 [dev] 及官方 Dev 派生物在 Hub 上常为 license=other，按产品约定标为不可商用。"""
    rid = repo_id.lower().replace("_", "-")
    if "schnell" in rid:
        return False
    if "depthpro" in rid or ("apple/" in rid and "depth" in rid):
        return True
    needles = (
        "flux.1-dev",
        "flux.1-kontext-dev",
        "flux.1-krea-dev",
        "flux.1-canny-dev",
        "flux.1-depth-dev",
        "flux.1-fill-dev",
        "flux.1-redux-dev",
        "flux.1-depth-dev-lora",
        "canny-dev-lora",
        "flux-uncensored",
    )
    return any(n in rid for n in needles)


def main() -> int:
    ap = argparse.ArgumentParser(description="Infer commercial_use_allowed from HF/ModelScope metadata")
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--dry-run", action="store_true", help="Print table only (default if neither flag)")
    ap.add_argument("--write", action="store_true", help="Write registry JSON with inferred values")
    ap.add_argument("--force", action="store_true", help="Overwrite existing true/false when --write")
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        help="HF Hub API base (e.g. https://hf-mirror.com)",
    )
    args = ap.parse_args()

    if not args.write and not args.dry_run:
        args.dry_run = True

    data = json.loads(args.registry.read_text(encoding="utf-8"))
    models = data.get("models")
    if not isinstance(models, dict):
        print("Invalid registry: missing models", file=sys.stderr)
        return 2

    rows: List[Tuple[str, Optional[bool], str, str]] = []

    for mid, rec in sorted(models.items()):
        if not isinstance(rec, dict):
            continue
        source = str(rec.get("source") or "huggingface").strip().lower()
        repo_id, vk = _repo_from_record(rec)
        if not repo_id:
            rows.append((mid, None, "skip", "no repo_id on default version"))
            continue

        licenses: List[str] = []
        prov = ""

        if source == "modelscope":
            ms_lic, err = _fetch_ms_license(repo_id)
            prov = f"modelscope:{repo_id}"
            if ms_lic:
                licenses.append(ms_lic)
            # 魔搭缺失 / other / 报错时，用 HF 同名仓库补 license 标签
            ms_token = _norm_license_token(ms_lic) if ms_lic else ""
            need_hf = (
                err is not None
                or not licenses
                or ms_token in ("other", "", "none")
            )
            if need_hf:
                hf_payload = _fetch_hf_model(repo_id, args.hf_endpoint, args.timeout)
                if hf_payload:
                    if " + hf:" not in prov:
                        prov += f" + hf:{repo_id}"
                    licenses.extend(_collect_hf_licenses(hf_payload))
            if not licenses:
                detail = f"modelscope: {err or 'no License'}; hf: no payload"
                rows.append((mid, None, "fail", detail))
                continue
        else:
            payload = _fetch_hf_model(repo_id, args.hf_endpoint, args.timeout)
            prov = f"hf:{repo_id}"
            if not payload:
                rows.append((mid, None, "fail", f"hf fetch failed ({args.hf_endpoint})"))
                continue
            licenses = _collect_hf_licenses(payload)

        inferred = _infer_from_license_strings(licenses)
        if inferred is None and _policy_false_commercial(repo_id):
            inferred = False

        lic_preview = ",".join(licenses[:5]) if licenses else "(none)"
        if inferred is None:
            rows.append((mid, None, "unknown", f"{prov} licenses={lic_preview}"))
        elif inferred is True:
            rows.append((mid, True, "true", f"{prov} licenses={lic_preview}"))
        else:
            rows.append((mid, False, "false", f"{prov} licenses={lic_preview}"))

    # print report
    for mid, inf, status, detail in rows:
        print(f"{mid}\t{status}\t{inf}\t{detail}")

    if not args.write:
        return 0

    # apply
    changed = 0
    for mid, inf, status, _detail in rows:
        if inf is None:
            continue
        rec = models.get(mid)
        if not isinstance(rec, dict):
            continue
        cur = rec.get("commercial_use_allowed")
        if cur in (True, False) and not args.force:
            continue
        if rec.get("commercial_use_allowed") != inf:
            rec["commercial_use_allowed"] = inf
            changed += 1

    args.registry.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.registry} (touched {changed} models).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
