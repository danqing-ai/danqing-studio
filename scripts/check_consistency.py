#!/usr/bin/env python3
"""轻量一致性检查：注册表 engine、任务路由顺序、前端 i18n nav 键。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "default_config" / "models_registry.json"
KNOWN = {"danqing-image", "danqing-video", "danqing-audio"}
REGISTRY_IMAGE_ACTION_KEYS = frozenset({"create", "rewrite", "retouch", "extend", "upscale"})
REGISTRY_VIDEO_ACTION_KEYS = frozenset({"create", "animate", "upscale"})
REGISTRY_AUDIO_ACTION_KEYS = frozenset({"create", "cover", "repaint"})
I18N_ZH = ROOT / "frontend" / "src" / "locales" / "zh.json"
I18N_EN = ROOT / "frontend" / "src" / "locales" / "en.json"
TASKS_ROUTES = ROOT / "backend" / "api" / "routes" / "tasks.py"
IMAGES_ROUTES = ROOT / "backend" / "api" / "routes" / "images.py"
VIDEOS_ROUTES = ROOT / "backend" / "api" / "routes" / "videos.py"
PRESETS_JSON = ROOT / "default_config" / "presets.json"
TASK_KINDS_PY = ROOT / "backend" / "core" / "task_kinds.py"
ASSETS_ROUTES = ROOT / "backend" / "api" / "routes" / "assets.py"
GALLERY_ROUTES = ROOT / "backend" / "api" / "routes" / "gallery.py"
INTERFACES_PY = ROOT / "backend" / "core" / "interfaces.py"
MEDIA_INTERFACES_PY = ROOT / "backend" / "core" / "media_interfaces.py"
MODELS_REGISTRY_JSON = ROOT / "default_config" / "models_registry.json"
MAKEFILE = ROOT / "Makefile"
ENGINE_GOVERNANCE_WORKFLOW = ROOT / ".github" / "workflows" / "engine-governance.yml"


def _load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_text(path):
    return path.read_text(encoding="utf-8")


def main():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    failures = []

    # =========================================================================
    # 1. 注册表 engines
    # =========================================================================
    data = _load_json(REG)
    engines = set(data.get("engines", {}).keys())
    unknown = engines - KNOWN
    if unknown:
        failures.append(f"Unknown engines: {unknown}")

    # =========================================================================
    # 2. models 与 actions
    # =========================================================================
    from backend.core.registry_profiles import validate_registry_document

    failures.extend(validate_registry_document(data))

    models = data.get("models", {})
    for m in models.values():
        engine = m.get("engine", "")
        actions_raw = m.get("actions", {})
        if isinstance(actions_raw, dict):
            actions = set(actions_raw.keys())
        elif isinstance(actions_raw, list):
            actions = set(actions_raw)
        else:
            actions = set()
        if engine.startswith("danqing-image"):
            invalid = actions - REGISTRY_IMAGE_ACTION_KEYS
        elif engine.startswith("danqing-video"):
            invalid = actions - REGISTRY_VIDEO_ACTION_KEYS
        elif engine.startswith("danqing-audio"):
            invalid = actions - REGISTRY_AUDIO_ACTION_KEYS
        else:
            invalid = set()
        if invalid:
            failures.append(f"Invalid actions for {m.get('id', '?')}: {invalid}")

    for model_id, m in models.items():
        if m.get("recommended") is True and m.get("commercial_use_allowed") is not True:
            failures.append(
                f"Model '{model_id}': recommended=true requires commercial_use_allowed=true"
            )
        fam = m.get("family")
        if fam is None or (isinstance(fam, str) and not str(fam).strip()):
            failures.append(f"Model '{model_id}': missing required non-empty 'family' field")

    # =========================================================================
    # 3. 预设 media_scope 校验
    # =========================================================================
    presets = _load_json(PRESETS_JSON)
    for name, preset in presets.items():
        if "applies_to" not in preset:
            failures.append(f"Preset '{name}' missing 'applies_to' field")
        if "media_scope" not in preset:
            failures.append(f"Preset '{name}' missing 'media_scope' field")

    # =========================================================================
    # 4. 路由顺序：tasks.py 中 /stream 必须晚于 /{id}/logs
    # =========================================================================
    tasks_body = _load_text(TASKS_ROUTES)
    stream_pos = tasks_body.find("/stream")
    logs_pos = tasks_body.find("/logs")
    if stream_pos != -1 and logs_pos != -1 and stream_pos < logs_pos:
        failures.append("FAIL: tasks.py /stream route must be defined after /{id}/logs")

    # =========================================================================
    # 5. images.py: create/edit/upscale 顺序
    # =========================================================================
    images_body = _load_text(IMAGES_ROUTES)
    gen_pos = images_body.find("/generations")
    edit_pos = images_body.find("/edits")
    upscale_pos = images_body.find("/upscales")
    if not (gen_pos < edit_pos < upscale_pos):
        failures.append("FAIL: images.py routes order must be generations/edits/upscales")

    # =========================================================================
    # 6. videos.py: create/edit/upscale 顺序
    # =========================================================================
    videos_body = _load_text(VIDEOS_ROUTES)
    vgen_pos = videos_body.find("/generations")
    vedit_pos = videos_body.find("/edits")
    vupscale_pos = videos_body.find("/upscales")
    if vgen_pos == -1 or vedit_pos == -1 or not (vgen_pos < vedit_pos):
        failures.append("FAIL: videos.py routes order must be generations before edits")
    if vupscale_pos != -1 and not (vedit_pos < vupscale_pos):
        failures.append("FAIL: videos.py /upscales must follow /edits when present")

    # =========================================================================
    # 7. 国际化键一致性
    # =========================================================================
    if I18N_ZH.exists() and I18N_EN.exists():
        zh = _load_json(I18N_ZH)
        en = _load_json(I18N_EN)
        zh_nav = zh.get("nav", {})
        en_nav = en.get("nav", {})
        for key in zh_nav:
            if key not in en_nav:
                failures.append(f"FAIL: i18n key 'nav.{key}' missing in en.json")
        for key in en_nav:
            if key not in zh_nav:
                failures.append(f"FAIL: i18n key 'nav.{key}' missing in zh.json")
    else:
        failures.append("FAIL: i18n JSON files not found in frontend/src/locales/")

    # =========================================================================
    # 8. task_kinds.py 与 models_registry.json 对齐
    # =========================================================================
    sys.path.insert(0, str(ROOT))
    from backend.core.task_kinds import ALL_KINDS, task_kind_for_registry_action

    registry = _load_json(MODELS_REGISTRY_JSON)
    for model_id, model in registry.get("models", {}).items():
        actions = model.get("actions", {})
        if not isinstance(actions, dict):
            continue
        media = model.get("media", "image")
        for action in actions:
            kind = task_kind_for_registry_action(media, action)
            if kind is None:
                failures.append(
                    f"FAIL: no task kind mapping for {model_id!r} media={media!r} action={action!r}"
                )
            elif kind not in ALL_KINDS:
                failures.append(f"FAIL: task kind {kind!r} not in ALL_KINDS")

    # =========================================================================
    # 9. interfaces.py 中 IImageEngine / IVideoEngine / IAudioEngine 能力声明
    # =========================================================================
    media_ifaces_body = _load_text(MEDIA_INTERFACES_PY)
    for iface in ("IImageEngine", "IVideoEngine", "IAudioEngine"):
        if iface not in media_ifaces_body:
            failures.append(f"FAIL: media_interfaces.py must define {iface}")
    if "async def generate" not in media_ifaces_body or "async def edit" not in media_ifaces_body:
        failures.append("FAIL: media_interfaces.py must define generate/edit methods")

    # =========================================================================
    # 10. assets.py 与 gallery.py 路由一致性
    # =========================================================================
    assets_body = _load_text(ASSETS_ROUTES)
    gallery_body = _load_text(GALLERY_ROUTES)
    if "/api/assets" not in assets_body:
        failures.append("FAIL: assets.py missing /api/assets route")
    if "/api/gallery" not in gallery_body:
        failures.append("FAIL: gallery.py missing /api/gallery route")

    # =========================================================================
    # 11. Engine governance target/workflow wiring
    # =========================================================================
    makefile_body = _load_text(MAKEFILE)
    if "check-engine-governance:" not in makefile_body:
        failures.append("FAIL: Makefile missing check-engine-governance target")
    if "verify-engine-stack:" not in makefile_body:
        failures.append("FAIL: Makefile missing verify-engine-stack target")
    if "check-engine-rules:" not in makefile_body:
        failures.append("FAIL: Makefile missing check-engine-rules target")
    if "check-models-registry-contracts" not in makefile_body:
        failures.append("FAIL: Makefile missing check-models-registry-contracts target")

    if not ENGINE_GOVERNANCE_WORKFLOW.exists():
        failures.append("FAIL: missing .github/workflows/engine-governance.yml")
    else:
        wf = _load_text(ENGINE_GOVERNANCE_WORKFLOW)
        if "make verify-engine-stack" not in wf:
            failures.append(
                "FAIL: engine-governance workflow must run `make verify-engine-stack`"
            )

    import subprocess

    rc = subprocess.call(
        [sys.executable, str(ROOT / "scripts" / "check_frontend_governance.py")],
        cwd=ROOT,
    )
    if rc != 0:
        failures.append(f"check_frontend_governance.py failed (exit {rc})")

    if failures:
        print(f"Consistency check failed ({len(failures)} errors):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("Consistency check OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())