"""Workspace dataset storage for LoRA training (mlx-examples ``train.jsonl`` format)."""

from __future__ import annotations

import json
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image

MIN_DATASET_IMAGES = 3
MAX_DATASET_IMAGES = 500


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def datasets_root(workspace_root: Path) -> Path:
    return workspace_root / "datasets"


def _dataset_dir(root: Path, dataset_id: str) -> Path:
    return root / dataset_id


def new_dataset_id() -> str:
    return "ds_" + secrets.token_hex(8)


def list_datasets(workspace_root: Path) -> list[dict[str, Any]]:
    root = datasets_root(workspace_root)
    if not root.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        meta_path = path / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        items.append(_summarize_dataset(path.name, path, meta))
    return items


def _summarize_dataset(dataset_id: str, path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    rows = _read_train_jsonl(path / "train.jsonl")
    return {
        "id": dataset_id,
        "name": meta.get("name") or dataset_id,
        "kind": meta.get("kind") or "concept",
        "trigger_word": meta.get("trigger_word") or "",
        "default_prompt": meta.get("default_prompt") or "",
        "nsfw": bool(meta.get("nsfw", False)),
        "image_count": len(rows),
        "created_at": meta.get("created_at"),
        "updated_at": meta.get("updated_at"),
    }


def _read_train_jsonl(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rows.append({"image": str(obj.get("image", "")), "prompt": str(obj.get("prompt", ""))})
    return rows


def _write_train_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"image": row["image"], "prompt": row["prompt"]}, ensure_ascii=False) + "\n")


def get_dataset(workspace_root: Path, dataset_id: str) -> dict[str, Any]:
    path = _dataset_dir(datasets_root(workspace_root), dataset_id)
    meta_path = path / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Dataset {dataset_id!r} not found")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = _read_train_jsonl(path / "train.jsonl")
    images = []
    for row in rows:
        img_rel = row["image"]
        img_path = path / img_rel
        images.append(
            {
                "file": img_rel,
                "prompt": row["prompt"],
                "exists": img_path.is_file(),
            }
        )
    summary = _summarize_dataset(dataset_id, path, meta)
    summary["images"] = images
    return summary


def create_dataset(
    workspace_root: Path,
    *,
    name: str,
    kind: str = "concept",
    trigger_word: str = "",
    default_prompt: str = "",
    nsfw: bool = False,
) -> dict[str, Any]:
    dataset_id = new_dataset_id()
    path = _dataset_dir(datasets_root(workspace_root), dataset_id)
    path.mkdir(parents=True, exist_ok=True)
    (path / "images").mkdir(exist_ok=True)
    meta = {
        "name": name.strip() or dataset_id,
        "kind": kind,
        "trigger_word": trigger_word.strip(),
        "default_prompt": default_prompt.strip(),
        "nsfw": nsfw,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_train_jsonl(path / "train.jsonl", [])
    return get_dataset(workspace_root, dataset_id)


def update_dataset_meta(workspace_root: Path, dataset_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    path = _dataset_dir(datasets_root(workspace_root), dataset_id)
    meta_path = path / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Dataset {dataset_id!r} not found")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("name", "kind", "trigger_word", "default_prompt", "nsfw"):
        if key in patch:
            meta[key] = patch[key]
    meta["updated_at"] = _now_iso()
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return get_dataset(workspace_root, dataset_id)


def add_dataset_images(
    workspace_root: Path,
    dataset_id: str,
    files: list[tuple[str, bytes]],
    *,
    default_prompt: str | None = None,
    per_file_prompts: list[str] | None = None,
) -> dict[str, Any]:
    path = _dataset_dir(datasets_root(workspace_root), dataset_id)
    if not (path / "meta.json").is_file():
        raise FileNotFoundError(f"Dataset {dataset_id!r} not found")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    prompt = (default_prompt if default_prompt is not None else meta.get("default_prompt") or "").strip()
    if not prompt and meta.get("trigger_word"):
        prompt = f"A photo of {meta['trigger_word']}"
    rows = _read_train_jsonl(path / "train.jsonl")
    images_dir = path / "images"
    images_dir.mkdir(exist_ok=True)
    for idx, (filename, data) in enumerate(files):
        safe_name = Path(filename).name or f"image_{len(rows):04d}.png"
        if not safe_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            safe_name = f"{safe_name}.png"
        dest = images_dir / safe_name
        counter = 0
        while dest.exists():
            counter += 1
            stem = Path(safe_name).stem
            dest = images_dir / f"{stem}_{counter}{Path(safe_name).suffix}"
        dest.write_bytes(data)
        rel = f"images/{dest.name}"
        row_prompt = prompt
        if per_file_prompts is not None and idx < len(per_file_prompts):
            row_prompt = str(per_file_prompts[idx] or prompt)
        rows.append({"image": rel, "prompt": row_prompt})
    if len(rows) > MAX_DATASET_IMAGES:
        raise ValueError(f"Dataset exceeds maximum of {MAX_DATASET_IMAGES} images")
    _write_train_jsonl(path / "train.jsonl", rows)
    meta["updated_at"] = _now_iso()
    (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return get_dataset(workspace_root, dataset_id)


def update_dataset_captions(
    workspace_root: Path,
    dataset_id: str,
    captions: list[dict[str, str]],
) -> dict[str, Any]:
    path = _dataset_dir(datasets_root(workspace_root), dataset_id)
    if not (path / "meta.json").is_file():
        raise FileNotFoundError(f"Dataset {dataset_id!r} not found")
    rows = _read_train_jsonl(path / "train.jsonl")
    by_file = {row["image"]: row for row in rows}
    for item in captions:
        file_key = item.get("file") or item.get("image")
        if not file_key or file_key not in by_file:
            continue
        by_file[file_key]["prompt"] = str(item.get("prompt") or "")
    _write_train_jsonl(path / "train.jsonl", list(by_file.values()))
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    meta["updated_at"] = _now_iso()
    (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return get_dataset(workspace_root, dataset_id)


def remove_dataset_image(workspace_root: Path, dataset_id: str, file_rel: str) -> dict[str, Any]:
    path = _dataset_dir(datasets_root(workspace_root), dataset_id)
    rows = _read_train_jsonl(path / "train.jsonl")
    rows = [r for r in rows if r["image"] != file_rel]
    img_path = path / file_rel
    if img_path.is_file():
        img_path.unlink()
    _write_train_jsonl(path / "train.jsonl", rows)
    return get_dataset(workspace_root, dataset_id)


def delete_dataset(workspace_root: Path, dataset_id: str) -> None:
    """Remove a training dataset directory from the workspace."""
    dataset_id = (dataset_id or "").strip()
    if not dataset_id or dataset_id.startswith("_") or "/" in dataset_id or "\\" in dataset_id:
        raise ValueError(f"Invalid dataset id {dataset_id!r}")
    path = _dataset_dir(datasets_root(workspace_root), dataset_id)
    if not (path / "meta.json").is_file():
        raise FileNotFoundError(f"Dataset {dataset_id!r} not found")
    shutil.rmtree(path)


def validate_dataset_for_training(workspace_root: Path, dataset_id: str) -> None:
    ds = get_dataset(workspace_root, dataset_id)
    count = int(ds.get("image_count") or 0)
    if count < MIN_DATASET_IMAGES:
        raise ValueError(
            f"Dataset needs at least {MIN_DATASET_IMAGES} images (has {count})"
        )


def _normalize_asset_id(ref: str) -> str:
    ref = ref.strip()
    if ref.startswith("asset:"):
        return ref[len("asset:") :]
    return ref


def import_dataset_from_assets(
    workspace_root: Path,
    dataset_id: str,
    asset_ids: list[str],
    *,
    resolve_asset_path: Callable[[str], Path],
    default_prompt: str | None = None,
    asset_captions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Copy gallery/canvas assets into a training dataset."""
    if not asset_ids:
        raise ValueError("asset_ids must not be empty")
    files: list[tuple[str, bytes]] = []
    prompts: list[str] = []
    path = _dataset_dir(datasets_root(workspace_root), dataset_id)
    if not (path / "meta.json").is_file():
        raise FileNotFoundError(f"Dataset {dataset_id!r} not found")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    fallback = (default_prompt if default_prompt is not None else meta.get("default_prompt") or "").strip()
    if not fallback and meta.get("trigger_word"):
        fallback = f"A photo of {meta['trigger_word']}"
    caps = asset_captions or {}
    for raw in asset_ids:
        aid = _normalize_asset_id(raw)
        if not aid:
            continue
        src = resolve_asset_path(aid)
        if not src.is_file():
            raise FileNotFoundError(f"Asset {aid!r} not found at {src}")
        suffix = src.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError(f"Asset {aid!r} is not an image ({suffix or 'unknown'})")
        cap = caps.get(raw) or caps.get(aid) or caps.get(f"asset:{aid}") or fallback
        files.append((src.name, src.read_bytes()))
        prompts.append(cap)
    if not files:
        raise ValueError("No valid image assets to import")
    return add_dataset_images(
        workspace_root,
        dataset_id,
        files,
        default_prompt=fallback,
        per_file_prompts=prompts,
    )


def bundled_dog6_example_dir(default_config_root: Path) -> Path:
    """Shipped DreamBooth dog6 example (Google DreamBooth, bundled in ``default_config/``)."""
    return default_config_root / "lora_examples" / "dog6"


def import_dog6_example(
    workspace_root: Path,
    *,
    bundled_root: Path | None = None,
    on_log: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Copy bundled DreamBooth dog6 example into the workspace datasets directory."""
    if bundled_root is None:
        raise RuntimeError("Dog6 example bundle path is not configured")
    bundled_root = Path(bundled_root)
    src_jsonl = bundled_root / "train.jsonl"
    if not src_jsonl.is_file():
        raise RuntimeError(f"Dog6 example bundle missing train.jsonl at {src_jsonl}")
    if on_log:
        on_log("info", f"Importing bundled Dog6 example from {bundled_root} …")

    ds = create_dataset(
        workspace_root,
        name="DreamBooth Dog6 (example)",
        kind="concept",
        trigger_word="sks dog",
        default_prompt="A photo of sks dog",
    )
    dataset_id = ds["id"]
    files: list[tuple[str, bytes]] = []
    for line in src_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rel = str(row.get("image") or "").strip()
        if not rel:
            continue
        src_img = bundled_root / rel
        if not src_img.is_file():
            src_img = bundled_root / "images" / Path(rel).name
        if not src_img.is_file():
            raise RuntimeError(f"Dog6 example bundle missing image {rel!r} (expected {src_img})")
        files.append((Path(rel).name, src_img.read_bytes()))
    if len(files) < MIN_DATASET_IMAGES:
        raise RuntimeError(
            f"Dog6 example bundle has {len(files)} images; need at least {MIN_DATASET_IMAGES}"
        )
    return add_dataset_images(workspace_root, dataset_id, files, default_prompt="A photo of sks dog")


def load_training_pairs(workspace_root: Path, dataset_id: str) -> list[tuple[Path, str]]:
    path = _dataset_dir(datasets_root(workspace_root), dataset_id)
    rows = _read_train_jsonl(path / "train.jsonl")
    pairs: list[tuple[Path, str]] = []
    for row in rows:
        img_path = path / row["image"]
        if img_path.is_file():
            pairs.append((img_path, row["prompt"]))
    return pairs


def resize_rgb_image(path: Path, resolution: tuple[int, int]) -> Any:
    import numpy as np

    img = Image.open(path).convert("RGB")
    img = img.resize(resolution, Image.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    return arr
