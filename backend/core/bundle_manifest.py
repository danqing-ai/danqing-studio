"""Bundle install manifest: scan, write, validate (fail loud at load time)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "bundle.manifest.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FamilyBundleContract:
    required: frozenset[str]
    optional: frozenset[str] = frozenset()


# Required component names align with scan_components keys.
FAMILY_BUNDLE_CONTRACTS: dict[str, FamilyBundleContract] = {
    "flux1": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder"}),
        optional=frozenset({"vae", "tokenizer"}),
    ),
    "flux2": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae"}),
        optional=frozenset({"tokenizer"}),
    ),
    "z_image": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae"}),
        optional=frozenset({"tokenizer"}),
    ),
    "qwen_image": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae"}),
        optional=frozenset({"tokenizer"}),
    ),
    "fibo": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae"}),
        optional=frozenset({"tokenizer"}),
    ),
    "seedvr2": FamilyBundleContract(
        required=frozenset({"transformer", "vae"}),
        optional=frozenset({"tokenizer"}),
    ),
    "ltx": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae"}),
        optional=frozenset({"tokenizer"}),
    ),
    "wan": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae"}),
        optional=frozenset({"tokenizer"}),
    ),
    "cogvideox": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae"}),
        optional=frozenset({"tokenizer"}),
    ),
    "hunyuan": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae"}),
        optional=frozenset({"tokenizer"}),
    ),
    "ace_step": FamilyBundleContract(
        required=frozenset({"transformer", "vae"}),
        optional=frozenset({"tokenizer"}),
    ),
    "heartmula": FamilyBundleContract(
        required=frozenset({"transformer"}),
        optional=frozenset({"vae", "tokenizer", "text_encoder"}),
    ),
}


def get_family_bundle_contract(family: str) -> FamilyBundleContract | None:
    return FAMILY_BUNDLE_CONTRACTS.get(family)


def require_family_bundle_contract(family: str) -> FamilyBundleContract:
    contract = get_family_bundle_contract(family)
    if contract is None:
        raise RuntimeError(
            f"No FamilyBundleContract registered for family={family!r}. "
            "Add an entry to bundle_manifest.FAMILY_BUNDLE_CONTRACTS before enabling bundle validation."
        )
    return contract


def _rel_path(bundle_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(bundle_root))
    except ValueError:
        return str(path)


def scan_components(bundle_root: Path) -> dict[str, list[str]]:
    """Scan bundle directory and classify files by component (convention-based)."""
    if not bundle_root.is_dir():
        raise FileNotFoundError(f"Bundle root not found: {bundle_root}")

    components: dict[str, list[str]] = {
        "transformer": [],
        "text_encoder": [],
        "vae": [],
        "tokenizer": [],
    }

    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file():
            continue
        rel_lower = _rel_path(bundle_root, path).lower()
        name_lower = path.name.lower()

        if name_lower == "tokenizer.json" or rel_lower.startswith("tokenizer/"):
            components["tokenizer"].append(_rel_path(bundle_root, path))
            continue

        if "text_encoder" in rel_lower or name_lower.startswith("text_encoder"):
            if path.suffix.lower() in (".safetensors", ".bin", ".json"):
                components["text_encoder"].append(_rel_path(bundle_root, path))
            continue

        if "/vae/" in f"/{rel_lower}/" or rel_lower.startswith("vae/") or name_lower.startswith("vae."):
            if path.suffix.lower() in (".safetensors", ".bin", ".json"):
                components["vae"].append(_rel_path(bundle_root, path))
            continue

        if path.suffix.lower() == ".safetensors":
            if "text_encoder" in rel_lower:
                components["text_encoder"].append(_rel_path(bundle_root, path))
            elif "vae" in rel_lower:
                components["vae"].append(_rel_path(bundle_root, path))
            elif (
                "transformer" in rel_lower
                or "diffusion" in name_lower
                or "unet" in name_lower
                or rel_lower.count("/") == 0
                or rel_lower.startswith("transformer/")
            ):
                components["transformer"].append(_rel_path(bundle_root, path))

    # Drop empty lists for cleaner manifest
    return {k: v for k, v in components.items() if v}


def missing_required_components(
    components: dict[str, list[str]],
    contract: FamilyBundleContract,
) -> list[str]:
    return sorted(name for name in contract.required if not components.get(name))


def build_manifest_payload(
    *,
    model_id: str,
    family: str,
    bundle_root: Path,
    contract: FamilyBundleContract | None = None,
) -> dict[str, Any]:
    components = scan_components(bundle_root)
    contract = contract or get_family_bundle_contract(family)
    missing: list[str] = []
    if contract is not None:
        missing = missing_required_components(components, contract)

    safetensors_count = sum(1 for p in bundle_root.rglob("*.safetensors"))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "family": family,
        "bundle_root": str(bundle_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "components": components,
        "detected": {
            "safetensors_count": safetensors_count,
            "weight_format": "safetensors" if safetensors_count else "unknown",
        },
        "status": "complete" if not missing else "incomplete",
    }
    if missing:
        payload["missing_components"] = missing
    return payload


def write_bundle_manifest(
    bundle_root: Path,
    *,
    model_id: str,
    family: str,
) -> Path:
    contract = get_family_bundle_contract(family)
    payload = build_manifest_payload(
        model_id=model_id,
        family=family,
        bundle_root=bundle_root,
        contract=contract,
    )
    out_path = bundle_root / MANIFEST_FILENAME
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def read_bundle_manifest(bundle_root: Path) -> dict[str, Any] | None:
    path = bundle_root / MANIFEST_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def assert_bundle_ready_for_family(
    bundle_root: Path,
    *,
    family: str,
    model_id: str,
) -> None:
    """Fail loud when required bundle components are missing."""
    if not bundle_root.is_dir():
        raise RuntimeError(
            f"Model {model_id!r} (family={family}): bundle directory missing at {bundle_root}"
        )

    contract = require_family_bundle_contract(family)
    manifest = read_bundle_manifest(bundle_root)
    if manifest is not None:
        components = manifest.get("components") or {}
        if not isinstance(components, dict):
            raise RuntimeError(
                f"Model {model_id!r}: invalid {MANIFEST_FILENAME} (components must be object)"
            )
        missing = missing_required_components(
            {k: list(v) if isinstance(v, list) else [] for k, v in components.items()},
            contract,
        )
    else:
        components = scan_components(bundle_root)
        missing = missing_required_components(components, contract)

    if missing:
        raise RuntimeError(
            f"Model {model_id!r} (family={family}): bundle at {bundle_root} is missing "
            f"required components: {missing}. Re-download or repair the model bundle."
        )


def bundle_component_status(bundle_root: Path, *, family: str) -> dict[str, Any] | None:
    """Component presence for download center; None when family has no contract."""
    contract = get_family_bundle_contract(family)
    if contract is None or not bundle_root.is_dir():
        return None

    manifest = read_bundle_manifest(bundle_root)
    if manifest is not None:
        raw = manifest.get("components") or {}
        components = {
            k: list(v) if isinstance(v, list) else []
            for k, v in raw.items()
            if isinstance(k, str)
        }
    else:
        components = scan_components(bundle_root)

    missing = missing_required_components(components, contract)
    tracked = sorted(contract.required | contract.optional)
    present = sorted(name for name in tracked if components.get(name))
    return {
        "complete": not missing,
        "missing": missing,
        "present": present,
        "components": {name: bool(components.get(name)) for name in tracked},
    }
