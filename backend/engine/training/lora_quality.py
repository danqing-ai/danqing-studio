"""Heuristic dataset health + post-training quality hints for LoRA DreamBooth."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from PIL import Image

from backend.engine.training import dataset_store

Level = Literal["good", "fair", "poor"]
Severity = Literal["info", "warning", "error"]

# Tunable thresholds (concept LoRA portraits; aligned with yz vs cyq observations).
_MIN_IMAGES_IDEAL = 10
_MIN_IMAGES_WARN = 6
_SHORT_EDGE_SMALL = 600
_SHORT_EDGE_TINY = 512
_MEDIAN_SHORT_EDGE_WARN = 720
_MEDIAN_SHORT_EDGE_POOR = 560
_VAL_GAP_WARN = 0.08
_VAL_REGRESSION_WARN = 0.08


def _hint(code: str, severity: Severity, **params: Any) -> dict[str, Any]:
    return {"code": code, "severity": severity, "params": params}


def _level_from_score(score: int, *, force_poor: bool = False, force_fair: bool = False) -> Level:
    if force_poor or score < 45:
        return "poor"
    if force_fair or score < 72:
        return "fair"
    return "good"


def _suspicious_filename(name: str) -> bool:
    lower = name.lower()
    return ".heic" in lower or ".heif" in lower or lower.endswith(".png.png")


def _probe_image(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"ok": False, "reason": "missing"}
    try:
        with Image.open(path) as img:
            img.load()
            w, h = img.size
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}
    short_edge = min(w, h)
    return {"ok": True, "width": w, "height": h, "short_edge": short_edge}


def analyze_dataset_health(workspace_root: Path, dataset_id: str) -> dict[str, Any]:
    """Scan dataset images for resolution / integrity issues before training."""
    ds = dataset_store.get_dataset(workspace_root, dataset_id)
    path = dataset_store.datasets_root(workspace_root) / dataset_id
    rows = ds.get("images") or []

    missing_count = 0
    broken_count = 0
    suspicious_count = 0
    small_600 = 0
    small_512 = 0
    empty_caption = 0
    short_edges: list[int] = []

    for row in rows:
        file_rel = str(row.get("file") or "")
        if not str(row.get("prompt") or "").strip():
            empty_caption += 1
        if _suspicious_filename(file_rel):
            suspicious_count += 1
        img_path = path / file_rel
        probe = _probe_image(img_path)
        if not probe.get("ok"):
            if probe.get("reason") == "missing":
                missing_count += 1
            else:
                broken_count += 1
            continue
        se = int(probe["short_edge"])
        short_edges.append(se)
        if se < _SHORT_EDGE_SMALL:
            small_600 += 1
        if se < _SHORT_EDGE_TINY:
            small_512 += 1

    total = len(rows)
    readable = len(short_edges)
    median_short = 0
    min_short = 0
    if short_edges:
        sorted_edges = sorted(short_edges)
        mid = len(sorted_edges) // 2
        median_short = sorted_edges[mid] if len(sorted_edges) % 2 else (
            (sorted_edges[mid - 1] + sorted_edges[mid]) // 2
        )
        min_short = sorted_edges[0]

    hints: list[dict[str, Any]] = []
    score = 100
    force_poor = False
    force_fair = False

    if missing_count:
        hints.append(_hint("missing_images", "error", count=missing_count, total=total))
        score -= min(40, missing_count * 15)
        force_poor = True
    if broken_count:
        hints.append(_hint("broken_images", "error", count=broken_count, total=total))
        score -= min(35, broken_count * 12)
        force_poor = True
    if suspicious_count:
        hints.append(_hint("suspicious_filenames", "warning", count=suspicious_count))
        score -= min(20, suspicious_count * 6)
        force_fair = True

    if total < _MIN_IMAGES_WARN:
        hints.append(_hint("too_few_images", "error", count=total, recommended=_MIN_IMAGES_IDEAL))
        score -= 25
        force_poor = True
    elif total < _MIN_IMAGES_IDEAL:
        hints.append(_hint("few_images", "warning", count=total, recommended=_MIN_IMAGES_IDEAL))
        score -= 10
        force_fair = True

    if readable and small_600:
        ratio = small_600 / readable
        if ratio >= 0.4 or small_600 >= 5:
            hints.append(
                _hint("many_small_600", "warning", count=small_600, total=readable, pct=round(ratio * 100))
            )
            score -= min(25, int(ratio * 30) + small_600)
            force_fair = True
    if readable and small_512:
        hints.append(_hint("many_small_512", "warning", count=small_512, total=readable))
        score -= min(20, small_512 * 4)
        if small_512 >= 3:
            force_fair = True

    if readable and median_short < _MEDIAN_SHORT_EDGE_POOR:
        hints.append(_hint("low_resolution_median", "error", median=median_short))
        score -= 25
        force_poor = True
    elif readable and median_short < _MEDIAN_SHORT_EDGE_WARN:
        hints.append(_hint("low_resolution_median", "warning", median=median_short))
        score -= 12
        force_fair = True

    if empty_caption:
        hints.append(_hint("empty_captions", "warning", count=empty_caption, total=total))
        score -= min(15, empty_caption * 3)

    if not hints and total >= _MIN_IMAGES_IDEAL and readable == total:
        hints.append(_hint("dataset_healthy", "info", count=total, median=median_short))

    score = max(0, min(100, score))
    level = _level_from_score(score, force_poor=force_poor, force_fair=force_fair)

    return {
        "level": level,
        "score": score,
        "stats": {
            "image_count": total,
            "readable_count": readable,
            "missing_count": missing_count,
            "broken_count": broken_count,
            "suspicious_filename_count": suspicious_count,
            "small_600_count": small_600,
            "small_512_count": small_512,
            "empty_caption_count": empty_caption,
            "median_short_edge": median_short,
            "min_short_edge": min_short,
        },
        "hints": hints,
    }


def _loss_points(loss_history: list[dict[str, Any]]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for row in loss_history:
        try:
            step = float(row.get("step") or 0)
            loss = float(row.get("loss") or 0)
        except (TypeError, ValueError):
            continue
        if step <= 0 or loss <= 0:
            continue
        pt: dict[str, float] = {"step": step, "loss": loss}
        if row.get("val_loss") is not None:
            try:
                pt["val_loss"] = float(row["val_loss"])
            except (TypeError, ValueError):
                pass
        out.append(pt)
    return sorted(out, key=lambda p: p["step"])


def analyze_training_quality(
    loss_history: list[dict[str, Any]],
    *,
    dataset_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize loss diagnostics without treating loss as a likeness score."""
    points = _loss_points(loss_history)
    hints: list[dict[str, Any]] = []
    score = 100
    force_fair = False

    metrics: dict[str, Any] = {
        "steps_logged": len(points),
        "initial_loss": None,
        "final_loss": None,
        "loss_drop_ratio": None,
        "final_val_loss": None,
        "best_val_loss": None,
    }

    if len(points) < 2:
        hints.append(_hint("insufficient_loss_data", "warning", steps=len(points)))
        return {
            "level": "fair",
            "score": 50,
            "metrics": metrics,
            "hints": hints,
            "dataset_health": dataset_health,
        }

    # Single-sample random-sigma loss is noisy; use short windows for diagnostics.
    window = max(1, min(5, len(points) // 4 or 1))
    initial = sum(p["loss"] for p in points[:window]) / window
    final = sum(p["loss"] for p in points[-window:]) / window
    drop_ratio = (initial - final) / initial if initial > 0 else 0.0
    metrics.update(
        {
            "initial_loss": round(initial, 4),
            "final_loss": round(final, 4),
            "loss_drop_ratio": round(drop_ratio, 4),
        }
    )

    hints.append(
        _hint(
            "loss_curve_diagnostic_only",
            "info",
            initial=round(initial, 4),
            final=round(final, 4),
        )
    )

    val_points = [p for p in points if "val_loss" in p]
    if val_points:
        final_val = val_points[-1]["val_loss"]
        metrics["final_val_loss"] = round(final_val, 4)
        best_val = min(p["val_loss"] for p in val_points)
        metrics["best_val_loss"] = round(best_val, 4)
        if final_val - final > _VAL_GAP_WARN:
            hints.append(
                _hint(
                    "val_loss_gap",
                    "warning",
                    val_loss=round(final_val, 4),
                    train_loss=round(final, 4),
                )
            )
            score -= 8
            force_fair = True
        if final_val - best_val > _VAL_REGRESSION_WARN:
            hints.append(
                _hint(
                    "val_loss_regressed",
                    "warning",
                    best_val_loss=round(best_val, 4),
                    final_val_loss=round(final_val, 4),
                )
            )
            score -= 8
            force_fair = True

    if dataset_health:
        ds_level = str(dataset_health.get("level") or "")
        if ds_level == "poor":
            hints.append(_hint("dataset_health_poor", "warning"))
            score -= 20
            force_fair = True
        elif ds_level == "fair":
            hints.append(_hint("dataset_health_fair", "info"))

    score = max(0, min(100, score))
    level = _level_from_score(score, force_fair=force_fair)

    return {
        "level": level,
        "score": score,
        "metrics": metrics,
        "hints": hints,
        "dataset_health": dataset_health,
    }
