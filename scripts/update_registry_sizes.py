#!/usr/bin/env python3
"""Probe remote model repos and refresh ``distribution.versions.*.size`` in models_registry.json.

Uses Hugging Face mirror / ModelScope file APIs (sizes match ``curl -I`` Content-Length on
resolve URLs). Respects ``allow_patterns``, ``bundle_repos``, and derived-version quantization
layout (install size after local MLX quant).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "default_config" / "models_registry.json"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
MS_ENDPOINT = "https://www.modelscope.cn"

_FAMILY_DEFAULT_LAYOUT: dict[str, str] = {
    "flux1": "diffusers_transformer",
    "flux2": "diffusers_transformer",
    "z_image": "diffusers_transformer",
    "qwen_image": "diffusers_transformer",
    "fibo": "diffusers_transformer",
    "seedvr2": "diffusers_transformer",
    "hunyuan": "diffusers_transformer",
    "ltx": "diffusers_transformer",
    "longcat": "diffusers_transformer",
    "wan": "wan_dit_shards",
    "diffrhythm": "dit_single_file",
    "ace_step": "dit_single_file",
}

_REPO_CACHE: dict[tuple[str, str], list[tuple[str, int]]] = {}


def _load_registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _http_json(url: str, *, timeout: float = 60.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "danqing-update-registry-sizes/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _list_hf_files(repo_id: str) -> list[tuple[str, int]]:
    key = ("hf", repo_id)
    if key in _REPO_CACHE:
        return _REPO_CACHE[key]
    url = f"{HF_ENDPOINT}/api/models/{repo_id}/tree/main?recursive=1"
    payload = _http_json(url)
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected HF tree payload for {repo_id}: {type(payload)}")
    files = [
        (str(item["path"]), int(item.get("size") or 0))
        for item in payload
        if isinstance(item, dict) and item.get("type") == "file"
    ]
    _REPO_CACHE[key] = files
    return files


def _list_ms_files(repo_id: str) -> list[tuple[str, int]]:
    key = ("ms", repo_id)
    if key in _REPO_CACHE:
        return _REPO_CACHE[key]
    url = (
        f"{MS_ENDPOINT}/api/v1/models/{repo_id}/repo/files"
        "?Revision=master&Recursive=true"
    )
    payload = _http_json(url)
    rows = payload.get("Data", {}).get("Files") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError(f"Unexpected ModelScope files payload for {repo_id}")
    files = [
        (str(item["Path"]), int(item.get("Size") or 0))
        for item in rows
        if isinstance(item, dict) and item.get("Type") == "blob"
    ]
    _REPO_CACHE[key] = files
    return files


def _list_repo_files(repo_id: str, source: str) -> list[tuple[str, int]]:
    src = source.strip().lower()
    if src == "modelscope":
        return _list_ms_files(repo_id)
    return _list_hf_files(repo_id)


def path_matches(path: str, pattern: str) -> bool:
    path = path.lstrip("./")
    pattern = pattern.lstrip("./")
    if fnmatch.fnmatchcase(path, pattern):
        return True
    if "**" not in pattern:
        return False
    if pattern.endswith("/**"):
        prefix = pattern[:-3]
        return path == prefix or path.startswith(prefix + "/")
    regex = "^" + re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*") + "$"
    return re.match(regex, path) is not None


def filter_files(
    files: list[tuple[str, int]],
    patterns: list[str] | None,
    *,
    catalog_type: str,
) -> list[tuple[str, int]]:
    if patterns:
        return [(p, s) for p, s in files if any(path_matches(p, pat) for pat in patterns)]
    if catalog_type in ("lora", "controlnet"):
        weights = [(p, s) for p, s in files if p.endswith((".safetensors", ".pth", ".bin", ".gguf"))]
        return weights if weights else files
    return files


def sum_files(files: list[tuple[str, int]]) -> int:
    return sum(size for _path, size in files)


def format_size(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0B"
    gib = num_bytes / (1024**3)
    if gib >= 10:
        return f"{round(gib)}GB"
    if gib >= 1:
        return f"{gib:.1f}GB"
    mib = num_bytes / (1024**2)
    if mib >= 10:
        return f"{round(mib)}MB"
    if mib >= 1:
        return f"{mib:.1f}MB"
    kib = num_bytes / 1024
    if kib >= 1:
        return f"{round(kib)}KB"
    return f"{num_bytes}B"


def _resolve_allow_patterns(
    ver_config: dict[str, Any],
    spec: dict[str, Any] | None,
    *,
    model_id: str,
) -> list[str] | None:
    if spec and isinstance(spec.get("allow_patterns"), list) and spec["allow_patterns"]:
        return [str(p) for p in spec["allow_patterns"]]
    if isinstance(ver_config.get("allow_patterns"), list) and ver_config["allow_patterns"]:
        return [str(p) for p in ver_config["allow_patterns"]]
    variant = ver_config.get("hunyuan_ms_variant")
    if variant:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from backend.services.hunyuan_ms_bundle import resolve_hunyuan_modelscope_allow_patterns

        resolved = resolve_hunyuan_modelscope_allow_patterns(ver_config, primary_spec=spec)
        if resolved:
            return resolved
    return None


def _repo_probe_specs(
    model_id: str,
    model_entry: dict[str, Any],
    ver_config: dict[str, Any],
) -> list[tuple[str, str, list[str] | None]]:
    catalog = model_entry.get("catalog") or {}
    default_source = str(catalog.get("source") or "huggingface").strip().lower()
    catalog_type = str(catalog.get("type") or "").lower()

    bundle = ver_config.get("bundle_repos")
    specs: list[tuple[str, str, list[str] | None]] = []
    if isinstance(bundle, list) and bundle:
        for item in bundle:
            if not isinstance(item, dict):
                continue
            repo_id = str(item.get("repo_id") or "").strip()
            if not repo_id:
                continue
            source = str(item.get("source") or ver_config.get("source") or default_source).strip().lower()
            patterns = _resolve_allow_patterns(ver_config, item, model_id=model_id)
            specs.append((repo_id, source, patterns))
        return specs

    repo_id = str(ver_config.get("repo_id") or "").strip()
    if not repo_id:
        return []
    source = str(ver_config.get("source") or default_source).strip().lower()
    patterns = _resolve_allow_patterns(ver_config, None, model_id=model_id)
    return [(repo_id, source, patterns)]


def _path_is_quant_target(path: str, *, layout: str) -> bool:
    if not path.endswith((".safetensors", ".pth", ".bin")):
        return False
    if layout == "diffusers_transformer":
        return path.startswith("transformer/") or (
            "/" not in path and path.endswith(".safetensors")
        )
    if layout == "wan_dit_shards":
        return (
            "diffusion_pytorch_model" in path
            or path.startswith("high_noise_model/")
            or path.startswith("low_noise_model/")
            or path.startswith("transformer/")
        )
    if layout == "dit_single_file":
        return path.endswith("model.safetensors") or path.endswith("/model.safetensors")
    return path.endswith(".safetensors")


def _estimate_derived_bytes(
    files: list[tuple[str, int]],
    *,
    family: str,
    quant: dict[str, Any],
) -> int:
    bits = int(quant.get("bits") or 16)
    ratio = max(bits, 1) / 16.0
    layout = str(quant.get("layout") or _FAMILY_DEFAULT_LAYOUT.get(family) or "diffusers_transformer")

    component_bits: dict[str, int] = {}
    for component in ("text_encoder", "text_encoder_2", "vae"):
        block = quant.get(component)
        if isinstance(block, dict) and block.get("bits") in (4, 8):
            component_bits[component] = int(block["bits"])

    total = 0
    for path, size in files:
        scaled = size
        for component, cbits in component_bits.items():
            if path.startswith(f"{component}/"):
                scaled = int(size * (cbits / 16.0))
                break
        else:
            if _path_is_quant_target(path, layout=layout):
                scaled = int(size * ratio)
        total += scaled
    return total


def probe_version_bytes(
    model_id: str,
    model_entry: dict[str, Any],
    ver_key: str,
    ver_config: dict[str, Any],
    *,
    versions: dict[str, Any],
) -> int:
    source_type = str(ver_config.get("source_type") or "full").lower()
    catalog_type = str((model_entry.get("catalog") or {}).get("type") or "").lower()

    if source_type == "derived":
        parent_key = str(ver_config.get("from_version") or "")
        parent = versions.get(parent_key) if parent_key else None
        if not isinstance(parent, dict):
            raise RuntimeError(f"{model_id}.{ver_key}: derived missing from_version {parent_key!r}")
        specs = _repo_probe_specs(model_id, model_entry, parent)
        if not specs:
            raise RuntimeError(f"{model_id}.{ver_key}: derived parent has no repo_id/bundle_repos")
        all_files: list[tuple[str, int]] = []
        for repo_id, source, patterns in specs:
            files = _list_repo_files(repo_id, source)
            all_files.extend(filter_files(files, patterns, catalog_type=catalog_type))
        family = str((model_entry.get("runtime") or {}).get("family") or "")
        quant = ver_config.get("quantization") or {}
        return _estimate_derived_bytes(all_files, family=family, quant=quant)

    total = 0
    for repo_id, source, patterns in _repo_probe_specs(model_id, model_entry, ver_config):
        files = _list_repo_files(repo_id, source)
        total += sum_files(filter_files(files, patterns, catalog_type=catalog_type))
    return total


def _should_skip(model_id: str, model_entry: dict[str, Any], ver_config: dict[str, Any]) -> bool:
    dist = model_entry.get("distribution") or {}
    if dist.get("stub_no_download"):
        return True
    if ver_config.get("stub_no_download"):
        return True
    specs = _repo_probe_specs(model_id, model_entry, ver_config)
    if ver_config.get("source_type") == "derived":
        parent_key = str(ver_config.get("from_version") or "")
        parent = (dist.get("versions") or {}).get(parent_key)
        if isinstance(parent, dict):
            specs = _repo_probe_specs(model_id, model_entry, parent)
    return not specs


def update_registry(
    data: dict[str, Any],
    *,
    dry_run: bool = False,
    only: set[str] | None = None,
    workers: int = 8,
) -> list[str]:
    changes: list[str] = []
    jobs: list[tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    models = data.get("models") or {}
    for model_id, model_entry in models.items():
        if only and model_id not in only:
            continue
        if not isinstance(model_entry, dict):
            continue
        dist = model_entry.get("distribution") or {}
        versions = dist.get("versions") or {}
        if not isinstance(versions, dict):
            continue
        for ver_key, ver_config in versions.items():
            if not isinstance(ver_config, dict):
                continue
            if _should_skip(model_id, model_entry, ver_config):
                continue
            jobs.append((model_id, ver_key, ver_config, model_entry, versions))

    def _run(job: tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]) -> tuple[str, str, int, str]:
        model_id, ver_key, ver_config, model_entry, versions = job
        num_bytes = probe_version_bytes(model_id, model_entry, ver_key, ver_config, versions=versions)
        return model_id, ver_key, num_bytes, format_size(num_bytes)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_run, job): job for job in jobs}
        for fut in as_completed(futures):
            model_id, ver_key, _ver_config, _model_entry, _versions = futures[fut]
            try:
                mid, vk, num_bytes, new_size = fut.result()
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                changes.append(f"ERROR {model_id}.{ver_key}: {exc}")
                continue
            ver_config = (models[mid]["distribution"]["versions"][vk])
            old_size = str(ver_config.get("size") or "")
            if old_size == new_size:
                continue
            if not dry_run:
                ver_config["size"] = new_size
                bundle = ver_config.get("bundle_repos")
                if isinstance(bundle, list) and bundle:
                    for item in bundle:
                        if not isinstance(item, dict):
                            continue
                        repo_id = str(item.get("repo_id") or "").strip()
                        if not repo_id:
                            continue
                        source = str(
                            item.get("source")
                            or ver_config.get("source")
                            or (models[mid].get("catalog") or {}).get("source")
                            or "huggingface"
                        ).strip().lower()
                        patterns = _resolve_allow_patterns(ver_config, item, model_id=mid)
                        try:
                            files = _list_repo_files(repo_id, source)
                            repo_bytes = sum_files(
                                filter_files(
                                    files,
                                    patterns,
                                    catalog_type=str((models[mid].get("catalog") or {}).get("type") or "").lower(),
                                )
                            )
                            item["size"] = format_size(repo_bytes)
                        except Exception:
                            pass
            changes.append(f"{mid}.{vk}: {old_size or '(none)'} -> {new_size} ({num_bytes / (1024**3):.2f} GiB)")

    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing registry")
    parser.add_argument("--model", action="append", default=[], help="Limit to model id(s)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel probe workers")
    args = parser.parse_args()

    data = _load_registry()
    only = set(args.model) if args.model else None
    changes = update_registry(data, dry_run=args.dry_run, only=only, workers=args.workers)

    errors = [c for c in changes if c.startswith("ERROR")]
    updates = [c for c in changes if not c.startswith("ERROR")]

    print(f"Probed updates: {len(updates)}; errors: {len(errors)}")
    for line in sorted(updates):
        print(line)
    for line in errors:
        print(line, file=sys.stderr)

    if not args.dry_run and updates:
        REGISTRY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {REGISTRY_PATH}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
