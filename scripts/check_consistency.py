#!/usr/bin/env python3
"""轻量一致性检查：注册表 engine、任务路由顺序、前端 i18n nav 键。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "config" / "models_registry.json"
KNOWN = {"danqing-image", "danqing-video", "danqing-audio"}
REGISTRY_IMAGE_ACTION_KEYS = frozenset({"create", "rewrite", "retouch", "extend", "upscale"})
REGISTRY_VIDEO_ACTION_KEYS = frozenset({"create", "animate"})
REGISTRY_AUDIO_ACTION_KEYS = frozenset({"create", "cover", "repaint"})
I18N = ROOT / "frontend" / "js" / "i18n.js"
INDEX_HTML = ROOT / "frontend" / "index.html"
APP_JS = ROOT / "frontend" / "js" / "app.js"
REGISTRY_PARAM_SCHEMA_JS = ROOT / "frontend" / "js" / "registry_param_schema.js"
REGISTRY_PARAMS_FORM_JS = ROOT / "frontend" / "js" / "components" / "RegistryParamsForm.js"
IMAGE_CREATE_PAGE_JS = ROOT / "frontend" / "js" / "components" / "ImageCreatePage.js"
VIDEO_CREATE_PAGE_JS = ROOT / "frontend" / "js" / "components" / "VideoCreatePage.js"
ASSET_PICKER_JS = ROOT / "frontend" / "js" / "components" / "AssetPicker.js"
ADAPTER_PICKER_JS = ROOT / "frontend" / "js" / "components" / "AdapterPicker.js"
IMAGE_EDITOR_JS = ROOT / "frontend" / "js" / "components" / "ImageEditor.js"
TASKS_ROUTES = ROOT / "backend" / "api" / "routes" / "tasks.py"
IMAGES_ROUTES = ROOT / "backend" / "api" / "routes" / "images.py"
VIDEOS_ROUTES = ROOT / "backend" / "api" / "routes" / "videos.py"
PRESETS_JSON = ROOT / "config" / "presets.json"
TASK_KINDS_PY = ROOT / "backend" / "core" / "task_kinds.py"
ASSETS_ROUTES = ROOT / "backend" / "api" / "routes" / "assets.py"
GALLERY_ROUTES = ROOT / "backend" / "api" / "routes" / "gallery.py"
INTERFACES_PY = ROOT / "backend" / "core" / "interfaces.py"
ASSET_STORE_PY = ROOT / "backend" / "persistence" / "asset_store.py"
MODELS_ROUTES = ROOT / "backend" / "api" / "routes" / "models.py"
API_JS = ROOT / "frontend" / "js" / "api.js"
MODELS_PAGE_JS = ROOT / "frontend" / "js" / "components" / "ModelsPage.js"
TASKS_STORE_JS = ROOT / "frontend" / "js" / "stores" / "tasks_store.js"
MEDIA_QUEUE_JS = ROOT / "frontend" / "js" / "composables" / "media_queue.js"
MEMORY_HINT_JS = ROOT / "frontend" / "js" / "composables" / "memory_hint.js"
TASK_STATUS_UI_JS = ROOT / "frontend" / "js" / "composables" / "task_status_ui.js"
MODEL_VERSION_VALUE_JS = ROOT / "frontend" / "js" / "composables" / "model_version_value.js"
STUDIO_NAV_JS = ROOT / "frontend" / "js" / "composables" / "studio_nav.js"
MIGRATE_PRESETS_SCRIPT = ROOT / "scripts" / "migrate_presets_mode_to_applies.py"
PLAN_7_5_SCHEMA_REVIEW_MD = ROOT / "docs" / "PLAN_7_5_SCHEMA_REVIEW.md"
REGISTRY_STORE_JS = ROOT / "frontend" / "js" / "stores" / "registry.js"
GALLERY_PAGE_JS = ROOT / "frontend" / "js" / "components" / "GalleryPage.js"
SETTINGS_PAGE_JS = ROOT / "frontend" / "js" / "components" / "SettingsPage.js"
SERVICES_PY = ROOT / "backend" / "services" / "services.py"
ADAPTERS_ROUTES = ROOT / "backend" / "api" / "routes" / "adapters.py"


def main() -> int:
    if not REG.exists():
        print("SKIP: no models_registry.json")
        return 0
    data = json.loads(REG.read_text(encoding="utf-8"))
    if data.get("schema_version") != 2:
        print("FAIL: models_registry.json must have schema_version: 2")
        return 1
    eng_block = data.get("engines") or {}
    if "danqing-image" not in eng_block or "danqing-video" not in eng_block or "danqing-audio" not in eng_block:
        print("FAIL: models_registry.json engines must include danqing-image, danqing-video, and danqing-audio")
        return 1
    for mid, cfg in (data.get("models") or {}).items():
        if not isinstance(cfg, dict):
            continue
        if not isinstance(cfg.get("actions"), dict):
            print("FAIL: model", mid, "must have actions dict (v2)")
            return 1
        media = cfg.get("media")
        if media not in ("image", "video", "audio"):
            print("FAIL: model", mid, "must have media: image|video|audio, got", repr(media))
            return 1
        if media == "image":
            allowed = REGISTRY_IMAGE_ACTION_KEYS
        elif media == "video":
            allowed = REGISTRY_VIDEO_ACTION_KEYS
        else:
            allowed = REGISTRY_AUDIO_ACTION_KEYS
        for action_key in cfg["actions"]:
            if action_key not in allowed:
                print(
                    "FAIL: model",
                    mid,
                    "registry action",
                    repr(action_key),
                    "not in plan verb set for",
                    media,
                    "(F4)",
                )
                return 1
        vers = cfg.get("versions")
        if not isinstance(vers, dict) or len(vers) < 1:
            print("FAIL: model", mid, "must have non-empty versions (registry install/delete are per-version)")
            return 1
    print("OK: registry schema v2 + media + F4 action keys + versions on all models")

    bad = []
    for mid, cfg in (data.get("models") or {}).items():
        eng = (cfg or {}).get("engine")
        if eng and eng not in KNOWN:
            bad.append((mid, eng))
    if bad:
        print("Unknown engine types:", bad)
        return 1
    print("OK: all model engines in", KNOWN)

    if TASKS_ROUTES.exists():
        text = TASKS_ROUTES.read_text(encoding="utf-8")
        root_pos = text.find('@router.get("")')
        list_pos = text.find('@router.get("/list")')
        dyn_pos = text.find('@router.get("/{task_id}")')
        if root_pos == -1 or dyn_pos == -1 or root_pos > dyn_pos:
            print("FAIL: tasks.py must declare GET \"\" (plan GET /api/tasks) before GET /{task_id}")
            return 1
        if list_pos == -1 or list_pos > dyn_pos:
            print("FAIL: tasks.py must declare GET /list before GET /{task_id}")
            return 1
        if '"/{task_id}/logs"' not in text or "get_task_logs" not in text:
            print("FAIL: tasks.py must expose GET /{task_id}/logs (plan paginated logs)")
            return 1
        if "@router.patch" not in text or "TaskPriorityPatch" not in text:
            print("FAIL: tasks.py must expose PATCH /{task_id} for priority (plan)")
            return 1
        if "kind:" not in text or "since:" not in text:
            print("FAIL: tasks.py list must accept kind/status/since filters")
            return 1
        print("OK: tasks route order")

    for label, path in (
        ("images", IMAGES_ROUTES),
        ("videos", VIDEOS_ROUTES),
    ):
        if path.exists():
            t = path.read_text(encoding="utf-8")
            if "/generations" not in t or "/edits" not in t:
                print(f"FAIL: {label}.py must expose generations and edits")
                return 1
            if label == "images" and "/upscales" not in t:
                print("FAIL: images.py must expose upscales")
                return 1
            print(f"OK: {label} routes present")

    if TASK_KINDS_PY.exists():
        tk = TASK_KINDS_PY.read_text(encoding="utf-8")
        if "IMAGE_GENERATION" not in tk or "ALL_KINDS" not in tk:
            print("FAIL: task_kinds.py must define IMAGE_* / VIDEO_* and ALL_KINDS")
            return 1
        print("OK: task_kinds module")

    if IMAGES_ROUTES.exists():
        im = IMAGES_ROUTES.read_text(encoding="utf-8")
        if "TK.IMAGE_GENERATION" not in im or "task_kinds" not in im:
            print("FAIL: images.py must submit using TK.* kinds from task_kinds")
            return 1
        print("OK: images routes use task_kinds")

    if VIDEOS_ROUTES.exists():
        vm = VIDEOS_ROUTES.read_text(encoding="utf-8")
        if "TK.VIDEO_GENERATION" not in vm or "task_kinds" not in vm:
            print("FAIL: videos.py must submit using TK.* kinds from task_kinds")
            return 1
        print("OK: videos routes use task_kinds")

    if IMAGES_ROUTES.exists() and VIDEOS_ROUTES.exists():
        im = IMAGES_ROUTES.read_text(encoding="utf-8")
        vm = VIDEOS_ROUTES.read_text(encoding="utf-8")
        for txt, path_needle, tk_needle in (
            (im, "/generations", "TK.IMAGE_GENERATION"),
            (im, "/edits", "TK.IMAGE_EDIT"),
            (im, "/upscales", "TK.IMAGE_UPSCALE"),
            (vm, "/generations", "TK.VIDEO_GENERATION"),
            (vm, "/edits", "TK.VIDEO_EDIT"),
        ):
            if path_needle not in txt or tk_needle not in txt:
                print("FAIL: F4 task kind ↔ route anchor", path_needle, tk_needle)
                return 1
        print("OK: F4 task kind ↔ media route anchors")

    if ASSETS_ROUTES.exists():
        ar = ASSETS_ROUTES.read_text(encoding="utf-8")
        if "{asset_id}/thumbnail" not in ar:
            print("FAIL: assets.py must expose GET /{asset_id}/thumbnail")
            return 1
        if "/reconcile" not in ar:
            print("FAIL: assets.py must expose POST /reconcile for disk vs DB reconciliation")
            return 1
        print("OK: assets thumbnail route + reconcile")
    if GALLERY_ROUTES.exists():
        gal = GALLERY_ROUTES.read_text(encoding="utf-8")
        if '@router.delete("/image")' not in gal or "delete_gallery_image" not in gal:
            print("FAIL: gallery.py must expose DELETE /image for Plan B4")
            return 1
        if '@router.get("/disk")' in gal or "get_gallery_disk_file" in gal:
            print("FAIL: gallery.py must not expose GET /disk (assets-only gallery)")
            return 1
        if "duration_seconds" not in gal or "GalleryItemResponse" not in gal:
            print("FAIL: gallery list_images must expose duration_seconds (Plan B2)")
            return 1
        print("OK: gallery DELETE /image, no /disk (Plan B4, assets-only)")
    if INTERFACES_PY.exists():
        iface = INTERFACES_PY.read_text(encoding="utf-8")
        if "gallery_mode" in iface:
            print("FAIL: AppSettings must not include gallery_mode (assets-only product)")
            return 1
        print("OK: AppSettings without gallery_mode")
    if ASSET_STORE_PY.exists():
        ast_src = ASSET_STORE_PY.read_text(encoding="utf-8")
        if "reconcile_disk_vs_db" not in ast_src:
            print("FAIL: asset_store.py must define reconcile_disk_vs_db")
            return 1
        if 'meta.get("num_frames")' not in ast_src or "duration is None" not in ast_src:
            print("FAIL: asset_store video branch must fall back duration from num_frames/fps (Plan B2)")
            return 1
        print("OK: asset_store reconcile_disk_vs_db")

    if SERVICES_PY.exists():
        sv = SERVICES_PY.read_text(encoding="utf-8")
        if "lora_adapter_picklist" not in sv:
            print("FAIL: SettingsService must define lora_adapter_picklist for adapters API")
            return 1
        print("OK: lora_adapter_picklist")

    if ADAPTERS_ROUTES.exists():
        ad = ADAPTERS_ROUTES.read_text(encoding="utf-8")
        if "list_adapters" not in ad or "for_model" not in ad:
            print("FAIL: adapters.py must expose list_adapters with for_model query")
            return 1
        print("OK: adapters route")

    if MODELS_ROUTES.exists():
        mp = MODELS_ROUTES.read_text(encoding="utf-8")
        if "/{model_id}/install" not in mp:
            print("FAIL: models.py must expose POST /{model_id}/install")
            return 1
        if "start_model_install" not in mp:
            print("FAIL: models.py must delegate install to download.start_model_install")
            return 1
        if "/{model_id}/versions/{version_key}" not in mp or "delete_registry_model_version" not in mp:
            print("FAIL: models.py must expose DELETE /{model_id}/versions/{version_key} for weight removal")
            return 1
        if "get_download_service" not in mp:
            print("FAIL: models.py must use get_download_service for delete_version")
            return 1
        if "/install-batch" not in mp or "install_registry_models_batch" not in mp:
            print("FAIL: models.py must expose POST /install-batch for registry batch installs")
            return 1
        print("OK: models install + install-batch + delete version routes")

    if API_JS.exists():
        aj = API_JS.read_text(encoding="utf-8")
        if "/api/models/" not in aj or "/install" not in aj or "async install(" not in aj:
            print("FAIL: api.js must expose models.install (POST /api/models/{id}/install)")
            return 1
        if "deleteVersion" not in aj or "/versions/" not in aj:
            print("FAIL: api.js must expose models.deleteVersion (DELETE /api/models/{id}/versions/{key})")
            return 1
        if "installBatch" not in aj or "/install-batch" not in aj:
            print("FAIL: api.js must expose models.installBatch (POST /api/models/install-batch)")
            return 1
        if "/api/download/model" in aj:
            print("FAIL: api.js must not call removed DELETE /api/download/model (use models.deleteVersion)")
            return 1
        if "/api/download/batch" in aj:
            print("FAIL: api.js must not call removed POST /api/download/batch (use models.installBatch)")
            return 1
        for needle in (
            "async cancel(taskId)",
            "async resume(taskId)",
            "async civitaiSearch(params)",
            "async startLoraDownload(url, filename)",
            "installProgressStreamUrl(taskId)",
        ):
            if needle not in aj:
                print("FAIL: api.js download section must include", needle)
                return 1
        if "logStreamUrl(taskId)" not in aj or "tasks:" not in aj:
            print("FAIL: api.js must expose tasks.logStreamUrl for task log SSE")
            return 1
        if (
            "listMediaTasks" not in aj
            or "getMediaTaskLogs" not in aj
            or "/logs?" not in aj
        ):
            print("FAIL: api.js must expose gen.listMediaTasks + gen.getMediaTaskLogs (plan tasks API)")
            return 1
        if "patchMediaTaskPriority" not in aj:
            print("FAIL: api.js must expose gen.patchMediaTaskPriority (PATCH /api/tasks/{id})")
            return 1
        if "/api/tasks?" not in aj:
            print("FAIL: api.js listMediaTasks must call plan GET /api/tasks (query on root)")
            return 1
        if "async metrics()" not in aj or "/api/system/metrics" not in aj:
            print("FAIL: api.js must expose api.system.metrics (GET /api/system/metrics)")
            return 1
        if "models: {\n        /** GET /api/models" not in aj or "async list(params = {})" not in aj:
            print("FAIL: api.js must expose api.models.list (GET /api/models with filters)")
            return 1
        if "window.api = api" not in aj:
            print("FAIL: api.js must assign window.api for stores / EventSource callers")
            return 1
        if "async urlToBlob(url)" not in aj:
            print("FAIL: api.js must expose gen.urlToBlob for create-page image / video blobs")
            return 1
        if "assetRowToGalleryItem" not in aj or "await api.gen.listAssets" not in aj:
            print("FAIL: api.js gallery.listImages must delegate to api.gen.listAssets (Plan C5)")
            return 1
        if "duration_seconds" not in aj:
            print("FAIL: api.js assetRowToGalleryItem must include duration_seconds (Plan B2)")
            return 1
        if "adapters:" not in aj or "/api/adapters" not in aj:
            print("FAIL: api.js must expose adapters.list (GET /api/adapters)")
            return 1
        if (
            "audios:" not in aj
            or "/api/audios/generations" not in aj
            or "/api/audios/edits" not in aj
            or "/api/audios/dubs" not in aj
        ):
            print("FAIL: api.js must expose audios.* for /api/audios/* (Plan I2 stub client)")
            return 1
        if "reconcileAssets" not in aj or "/api/assets/reconcile" not in aj:
            print("FAIL: api.js must expose gen.reconcileAssets (POST /api/assets/reconcile)")
            return 1
        if "/api/gallery/image" not in aj:
            print("FAIL: api.js gallery.deleteImage must use DELETE /api/gallery/image (Plan B4)")
            return 1
        if "/api/gallery/disk" in aj or "disk:" in aj:
            print("FAIL: api.js gallery must not reference /api/gallery/disk or disk: URLs (assets-only)")
            return 1
        if "expected asset:id" not in aj:
            print("FAIL: api.js gallery.getImageUrl must accept asset: paths only")
            return 1
        print("OK: api.js models + download + adapters + audios stubs + reconcile + gallery (asset: only)")

    if MODELS_PAGE_JS.exists():
        pg = MODELS_PAGE_JS.read_text(encoding="utf-8")
        if "api.models.install" not in pg:
            print("FAIL: ModelsPage must call api.models.install for registry model downloads")
            return 1
        if "api.models.deleteVersion" not in pg:
            print("FAIL: ModelsPage must call api.models.deleteVersion for registry version removal")
            return 1
        if "api.models.installBatch" not in pg:
            print("FAIL: ModelsPage must call api.models.installBatch for recommended batch installs")
            return 1
        if "fetch(`/api/download/model/" in pg or 'fetch(`/api/download/model/' in pg:
            print("FAIL: ModelsPage must not use legacy fetch /api/download/model for registry installs")
            return 1
        if "/api/download/batch" in pg:
            print("FAIL: ModelsPage must not call removed POST /api/download/batch")
            return 1
        if "fetch(" in pg:
            print("FAIL: ModelsPage must use api.* / axios via api.js, not raw fetch()")
            return 1
        for needle in (
            "api.download.listDownloads",
            "api.download.cancel",
            "api.download.resume",
            "api.download.civitaiSearch",
            "api.download.startLoraDownload",
            "api.download.installProgressStreamUrl",
        ):
            if needle not in pg:
                print("FAIL: ModelsPage must call", needle)
                return 1
        print("OK: ModelsPage uses api.models + api.download (no fetch)")

    if TASKS_STORE_JS.exists():
        ts = TASKS_STORE_JS.read_text(encoding="utf-8")
        if "w.api.tasks.logStreamUrl" not in ts:
            print("FAIL: tasks_store must use w.api.tasks.logStreamUrl for task log SSE")
            return 1
        if "w.api.gen.getQueue" not in ts:
            print("FAIL: tasks_store must prefer w.api.gen.getQueue for queue polling")
            return 1
        if "addEventListener('progress'" not in ts or "addEventListener('result'" not in ts:
            print("FAIL: tasks_store SSE must handle progress + result (plan §6.3)")
            return 1
        print("OK: tasks_store prefers api.tasks + api.gen.getQueue")

    if MEDIA_QUEUE_JS.exists():
        mq = MEDIA_QUEUE_JS.read_text(encoding="utf-8")
        if "DQMediaQueue" not in mq or "tasksForMedia" not in mq:
            print("FAIL: media_queue.js must define DQMediaQueue.tasksForMedia")
            return 1
        if "snapshotFullQueue" not in mq or "normalizeTaskRow" not in mq:
            print("FAIL: media_queue.js must export snapshotFullQueue + normalizeTaskRow for app.js drawer")
            return 1
        print("OK: composables/media_queue.js (plan C9)")
    else:
        print("FAIL: missing composables/media_queue.js")
        return 1

    if MEMORY_HINT_JS.exists():
        mh = MEMORY_HINT_JS.read_text(encoding="utf-8")
        if "DQMemoryHint" not in mh or "warnIfRisky" not in mh:
            print("FAIL: memory_hint.js must define DQMemoryHint.warnIfRisky (plan E4)")
            return 1
        print("OK: composables/memory_hint.js (plan E4)")
    else:
        print("FAIL: missing composables/memory_hint.js")
        return 1

    if TASK_STATUS_UI_JS.exists():
        tsu = TASK_STATUS_UI_JS.read_text(encoding="utf-8")
        if "DQTaskStatusUi" not in tsu or "tagType" not in tsu or "statusText" not in tsu:
            print("FAIL: task_status_ui.js must define DQTaskStatusUi.tagType + statusText")
            return 1
        print("OK: composables/task_status_ui.js")
    else:
        print("FAIL: missing composables/task_status_ui.js")
        return 1

    if MODEL_VERSION_VALUE_JS.exists():
        mvv = MODEL_VERSION_VALUE_JS.read_text(encoding="utf-8")
        if "DQModelVersionValue" not in mvv or "parse" not in mvv:
            print("FAIL: model_version_value.js must define DQModelVersionValue.parse")
            return 1
        print("OK: composables/model_version_value.js")
    else:
        print("FAIL: missing composables/model_version_value.js")
        return 1

    if STUDIO_NAV_JS.exists():
        sn = STUDIO_NAV_JS.read_text(encoding="utf-8")
        if "DQStudioNav" not in sn or "goSettings" not in sn or "goModels" not in sn:
            print("FAIL: studio_nav.js must define DQStudioNav.goSettings + goModels")
            return 1
        print("OK: composables/studio_nav.js")
    else:
        print("FAIL: missing composables/studio_nav.js")
        return 1

    if MIGRATE_PRESETS_SCRIPT.exists():
        mp = MIGRATE_PRESETS_SCRIPT.read_text(encoding="utf-8")
        if (
            "applies_to" not in mp
            or "MODE_TO_APPLIES" not in mp
            or "--dry-run" not in mp
            or "ensure_media_scope_inplace" not in mp
        ):
            print("FAIL: migrate_presets_mode_to_applies.py must map mode -> applies_to + media_scope (G1/G2)")
            return 1
        print("OK: scripts/migrate_presets_mode_to_applies.py (plan G1/G2)")
    else:
        print("FAIL: missing scripts/migrate_presets_mode_to_applies.py")
        return 1

    if REGISTRY_STORE_JS.exists():
        rs = REGISTRY_STORE_JS.read_text(encoding="utf-8")
        if "w.api.registry.getFull" not in rs:
            print("FAIL: registry store must prefer w.api.registry.getFull over raw /api/registry")
            return 1
        print("OK: RegistryStore prefers api.registry.getFull")

    if IMAGE_CREATE_PAGE_JS.exists():
        icx = IMAGE_CREATE_PAGE_JS.read_text(encoding="utf-8")
        if "api.gallery.getImageUrl(`asset:${pid}`)" not in icx:
            print("FAIL: ImageCreatePage must build completed preview URL via api.gallery.getImageUrl(asset:id)")
            return 1
        if "api.gen.urlToBlob" not in icx:
            print("FAIL: ImageCreatePage must use api.gen.urlToBlob for edit/control/ref image bytes")
            return 1
        if "fetch(" in icx:
            print("FAIL: ImageCreatePage must not use raw fetch() (use api.gen.urlToBlob)")
            return 1
        if "api.settings.getPresets" not in icx or "filteredPresets" not in icx:
            print("FAIL: ImageCreatePage must load presets via api.settings.getPresets (Plan G2)")
            return 1
    if VIDEO_CREATE_PAGE_JS.exists():
        vcx = VIDEO_CREATE_PAGE_JS.read_text(encoding="utf-8")
        if "api.gallery.getImageUrl(`asset:${pid}`)" not in vcx:
            print("FAIL: VideoCreatePage must build completed preview URL via api.gallery.getImageUrl(asset:id)")
            return 1
        if "api.gen.urlToBlob" not in vcx:
            print("FAIL: VideoCreatePage must use api.gen.urlToBlob for start-frame bytes")
            return 1
        if "fetch(" in vcx:
            print("FAIL: VideoCreatePage must not use raw fetch() (use api.gen.urlToBlob)")
            return 1
        if "api.settings.getPresets" not in vcx or "filteredPresets" not in vcx:
            print("FAIL: VideoCreatePage must load presets via api.settings.getPresets (Plan G2)")
            return 1
    if IMAGE_CREATE_PAGE_JS.exists() and VIDEO_CREATE_PAGE_JS.exists():
        print("OK: image/video create pages use api.gallery preview + api.gen.urlToBlob (no fetch)")

    if PRESETS_JSON.exists():
        presets = json.loads(PRESETS_JSON.read_text(encoding="utf-8"))
        for name, pr in presets.items():
            if not isinstance(pr, dict):
                print("FAIL: preset", name, "must be object")
                return 1
            if "mode" in pr:
                print("FAIL: preset", name, "must not use legacy mode field")
                return 1
            if not isinstance(pr.get("applies_to"), list) or not pr["applies_to"]:
                print("FAIL: preset", name, "must have non-empty applies_to array")
                return 1
            ms = pr.get("media_scope")
            if ms not in ("image", "video"):
                print("FAIL: preset", name, "must have media_scope image or video (Plan G2)")
                return 1
        print("OK: presets applies_to + media_scope schema")

    if not PLAN_7_5_SCHEMA_REVIEW_MD.is_file():
        print("FAIL: docs/PLAN_7_5_SCHEMA_REVIEW.md missing (Plan H1 vs §7.5)")
        return 1
    _h1 = PLAN_7_5_SCHEMA_REVIEW_MD.read_text(encoding="utf-8")
    if "V3TaskStore" not in _h1 or "SQLiteAssetStore" not in _h1:
        print("FAIL: PLAN_7_5_SCHEMA_REVIEW must name V3TaskStore and SQLiteAssetStore")
        return 1
    print("OK: docs/PLAN_7_5_SCHEMA_REVIEW.md (Plan H1)")

    if GALLERY_PAGE_JS.exists():
        gp = GALLERY_PAGE_JS.read_text(encoding="utf-8")
        if "api.gallery.listImages" not in gp:
            print("FAIL: GalleryPage must load gallery via api.gallery.listImages (Plan B3)")
            return 1
        if "formatVideoDuration" not in gp or "duration_seconds" not in gp:
            print("FAIL: GalleryPage must show video duration from duration_seconds (Plan B2)")
            return 1
        print("OK: GalleryPage uses gallery.listImages")

    if I18N.exists():
        body = I18N.read_text(encoding="utf-8").replace("\r\n", "\n")
        if "gallery: '图库',\n            models: '模型'," not in body:
            print("FAIL: i18n.js zh nav must include models after gallery")
            return 1
        if "gallery: 'Gallery',\n            models: 'Models'," not in body:
            print("FAIL: i18n.js en nav must include models after gallery")
            return 1
        if "assetPicker:" not in body or "needImage:" not in body:
            print("FAIL: i18n.js must define assetPicker (upload / library / needImage)")
            return 1
        if body.count("surfaceGeneration:") < 2:
            print("FAIL: i18n.js zh+en must define action.image.surfaceGeneration (plan D1)")
            return 1
        if body.count("retouchDesc:") < 2:
            print("FAIL: i18n.js zh+en must define action.image.retouchDesc (plan D1)")
            return 1
        if "文生视频" not in body or "Text-to-video" not in body:
            print("FAIL: i18n.js must define action.video.create (zh + en) for plan D1")
            return 1
        if "图生视频" not in body or "Image-to-video" not in body:
            print("FAIL: i18n.js must define action.video.animate (zh + en) for plan D1")
            return 1
        if body.count("startImage:") < 2:
            print("FAIL: i18n.js zh+en must define action.video.startImage")
            return 1
        if body.count("studio: {") < 2:
            print("FAIL: i18n.js zh+en must define studio block (plan D shared UI)")
            return 1
        if "logsEmpty: '暂无日志'" not in body or "logsEmpty: 'No logs'" not in body:
            print("FAIL: i18n studio.logsEmpty must exist in zh and en")
            return 1
        if body.count("modelNotReady: '模型 {name} 未下载'") < 1:
            print("FAIL: i18n studio.modelNotReady (zh) required for shared model alerts")
            return 1
        # en studio block grew; scan a wide prefix of the locale tail after the 2nd "studio: {"
        if "cancelled: 'Cancelled'" not in body.split("studio: {", 2)[-1][:12000]:
            print("FAIL: i18n en studio block must include task status keys (e.g. cancelled)")
            return 1
        if "submitOomHint:" not in body or "modelGb" not in body or "refGb" not in body:
            print("FAIL: i18n studio.submitOomHint (modelGb, refGb) required for Plan E4")
            return 1
        if "durationLabel: '时长'" not in body or "durationLabel: 'Duration'" not in body:
            print("FAIL: i18n gallery.durationLabel zh+en required (Plan B2)")
            return 1
        if body.count("durationSecs:") < 2:
            print("FAIL: i18n gallery.durationSecs must exist in zh and en (Plan B2)")
            return 1
        if "needControlImage: '使用 ControlNet 时请上传控制图'" not in body:
            print("FAIL: i18n studio.needControlImage (zh) required for ControlNet validation toast")
            return 1
        if "needControlImage: 'Please upload a control image when using ControlNet'" not in body:
            print("FAIL: i18n studio.needControlImage (en) required")
            return 1
        if body.count("genComplete: '生成完成!'") < 1:
            print("FAIL: i18n studio.genComplete (zh) required for shared generation logs")
            return 1
        if "prompt: '提示词'" not in body or "negativePrompt: '负面提示词'" not in body:
            print("FAIL: i18n studio.prompt / studio.negativePrompt (zh) required (plan D)")
            return 1
        if "switchModel: '已切换模型: {name} ({version})'" not in body:
            print("FAIL: i18n studio.switchModel (zh) required (plan D)")
            return 1
        if "switchModel: 'Switched to model: {name} ({version})'" not in body:
            print("FAIL: i18n studio.switchModel (en) required (plan D)")
            return 1
        if "queueSetHigh:" not in body or "priorityUpdated:" not in body:
            print("FAIL: i18n studio.queueSetHigh + studio.priorityUpdated (PATCH queue UI)")
            return 1
        print("OK: i18n nav.models labels + assetPicker + plan D action.* + studio.*")

    if (
        REGISTRY_PARAM_SCHEMA_JS.exists()
        and REGISTRY_PARAMS_FORM_JS.exists()
        and IMAGE_CREATE_PAGE_JS.exists()
    ):
        ic = IMAGE_CREATE_PAGE_JS.read_text(encoding="utf-8")
        if "registry-params-form" not in ic:
            print("FAIL: ImageCreatePage must use <registry-params-form> for advanced params")
            return 1
        if "RegistryParamSchema" not in ic:
            print("FAIL: ImageCreatePage must call RegistryParamSchema for defaults / deviation")
            return 1
        if REGISTRY_PARAMS_FORM_JS.exists():
            rf = REGISTRY_PARAMS_FORM_JS.read_text(encoding="utf-8")
            if "<adapter-picker" not in rf:
                print("FAIL: RegistryParamsForm must render AdapterPicker for LoRA row")
                return 1
            if "controlRecentGallery" not in rf or "control-asset-pick" not in rf:
                print(
                    "FAIL: RegistryParamsForm must use AssetPicker for ControlNet "
                    "(controlRecentGallery + control-asset-pick)"
                )
                return 1
            if "<asset-picker" not in rf:
                print("FAIL: RegistryParamsForm must render <asset-picker> for ControlNet")
                return 1
            if "studio.resolution" not in rf or "studio.controlNet" not in rf:
                print("FAIL: RegistryParamsForm must use studio.* for shared param labels (plan D)")
                return 1
        if INDEX_HTML.exists():
            idx = INDEX_HTML.read_text(encoding="utf-8")
            if "registry_store.js" in idx:
                print("FAIL: index.html must not reference removed registry_store.js (use stores/registry.js)")
                return 1
            if "stores/tasks_store.js" not in idx or "stores/registry.js" not in idx:
                print("FAIL: index.html must load stores/tasks_store.js and stores/registry.js")
                return 1
            if "composables/media_queue.js" not in idx:
                print("FAIL: index.html must load composables/media_queue.js (plan C9)")
                return 1
            if "composables/memory_hint.js" not in idx:
                print("FAIL: index.html must load composables/memory_hint.js (plan E4)")
                return 1
            if "composables/task_status_ui.js" not in idx or "composables/model_version_value.js" not in idx:
                print("FAIL: index.html must load task_status_ui.js and model_version_value.js before create pages")
                return 1
            if "composables/studio_nav.js" not in idx:
                print("FAIL: index.html must load composables/studio_nav.js before create pages")
                return 1
            if not (
                idx.find("stores/tasks_store.js")
                < idx.find("composables/media_queue.js")
                < idx.find("composables/memory_hint.js")
                < idx.find("composables/task_status_ui.js")
                < idx.find("composables/model_version_value.js")
                < idx.find("composables/studio_nav.js")
                < idx.find("ImageCreatePage.js")
            ):
                print(
                    "FAIL: index.html composables chain: tasks_store → media_queue → memory_hint → "
                    "task_status_ui → model_version_value → studio_nav → ImageCreatePage"
                )
                return 1
            if idx.find("js/api.js") > idx.find("stores/tasks_store.js"):
                print("FAIL: index.html must load api.js before stores/tasks_store.js")
                return 1
            if "AdapterPicker.js" not in idx or "RegistryParamsForm.js" not in idx:
                print("FAIL: index.html must load AdapterPicker.js and RegistryParamsForm.js")
                return 1
            if idx.find("AdapterPicker.js") > idx.find("RegistryParamsForm.js"):
                print("FAIL: index.html must load AdapterPicker.js before RegistryParamsForm.js")
                return 1
            if "AssetPicker.js" not in idx:
                print("FAIL: index.html must load AssetPicker.js")
                return 1
            if "ImageEditor.js" not in idx:
                print("FAIL: index.html must load ImageEditor.js")
                return 1
            if idx.find("AssetPicker.js") > idx.find("ImageEditor.js"):
                print("FAIL: index.html must load AssetPicker.js before ImageEditor.js (plan C4/C6)")
                return 1
            if idx.find("ImageEditor.js") > idx.find("ImageCreatePage.js"):
                print("FAIL: index.html must load ImageEditor.js before ImageCreatePage.js")
                return 1
            if idx.find("AssetPicker.js") > idx.find("RegistryParamsForm.js"):
                print("FAIL: index.html must load AssetPicker.js before RegistryParamsForm.js")
                return 1
            if "registry_param_schema.js" not in idx or "RegistryParamsForm.js" not in idx:
                print("FAIL: index.html must load registry_param_schema.js before ImageCreatePage")
                return 1
            if "showGlobalQueueDrawer" not in idx or "globalQueueCount" not in idx:
                print("FAIL: index.html must include global task queue drawer + badge (plan TopNav)")
                return 1
            # TaskDrawer is inline in index.html (el-drawer Teleport to body; wrapping in component causes emitsOptions errors)
            if "dq-task-queue-drawer" not in idx:
                print("FAIL: index.html el-drawer must use class dq-task-queue-drawer for dark theme")
                return 1
            # TopNav shell component
            TOPNAV_JS = Path("frontend/js/components/shell/TopNav.js")
            if not TOPNAV_JS.exists():
                print("FAIL: TopNav shell component missing (plan §9)")
                return 1
            if "TopNav.js" not in idx:
                print("FAIL: index.html must load shell/TopNav.js")
                return 1
            # Router
            ROUTER_JS = Path("frontend/js/router.js")
            if not ROUTER_JS.exists():
                print("FAIL: router.js missing (plan §9)")
                return 1
            if "router.js" not in idx:
                print("FAIL: index.html must load router.js")
                return 1
            if "setQueuedPriority" not in idx or "studio.queueSetHigh" not in idx:
                print("FAIL: index.html queue drawer must expose PATCH priority (setQueuedPriority + i18n)")
                return 1
        if APP_JS.exists():
            app_body = APP_JS.read_text(encoding="utf-8")
            if "ModelsPage" not in app_body or "DownloadPage" in app_body:
                print("FAIL: app.js must register ModelsPage (not legacy DownloadPage)")
                return 1
            if "AdapterPicker" not in app_body:
                print("FAIL: app.js must register AdapterPicker")
                return 1
            if "AssetPicker" not in app_body:
                print("FAIL: app.js must register AssetPicker")
                return 1
            if "RegistryParamsForm" not in app_body:
                print("FAIL: app.js must register RegistryParamsForm")
                return 1
            # Shell components (plan §9)
            if "app.component('TopNav'" not in app_body:
                print("FAIL: app.js must register TopNav shell component (TaskDrawer is inline in index.html)")
                return 1
            if "DQRouter" not in app_body:
                print("FAIL: app.js must integrate DQRouter (plan §9 hash-based router)")
                return 1
            ic_pos = app_body.find("app.component('ImageCreatePage'")
            ie_pos = app_body.find("app.component('ImageEditor'")
            if ie_pos == -1 or ic_pos == -1 or ie_pos > ic_pos:
                print("FAIL: app.js must register ImageEditor before ImageCreatePage")
                return 1
            if "ensureQueuePoller" not in app_body or "releaseQueuePoller" not in app_body:
                print("FAIL: app.js must start/stop TasksStore queue poller at app root")
                return 1
            if "snapshotFullQueue" not in app_body:
                print("FAIL: app.js global queue must use DQMediaQueue.snapshotFullQueue (dedupe row norm)")
                return 1
            if "open-global-task-queue" not in app_body:
                print("FAIL: app.js must listen for open-global-task-queue to open drawer")
                return 1
            if "patchMediaTaskPriority" not in app_body or "setQueuedPriority" not in app_body:
                print("FAIL: app.js must wire gen.patchMediaTaskPriority via setQueuedPriority (plan PATCH tasks)")
                return 1
        if ASSET_PICKER_JS.exists():
            ap = ASSET_PICKER_JS.read_text(encoding="utf-8")
            if "api.gen.listAssets" not in ap or "api.gen.uploadAsset" not in ap:
                print("FAIL: AssetPicker must use api.gen.listAssets and api.gen.uploadAsset")
                return 1
            if "studio.uploadFailed" not in ap:
                print("FAIL: AssetPicker must use studio.uploadFailed for errors (plan D)")
                return 1
            if "asset-picker__recent-grid" not in ap:
                print("FAIL: AssetPicker must use layout classes (asset-picker__*) for ref-image-placeholder")
                return 1
            print("OK: AssetPicker.js")
        if ADAPTER_PICKER_JS.exists():
            adp = ADAPTER_PICKER_JS.read_text(encoding="utf-8")
            if "studio.loraLabel" not in adp or "studio.noLora" not in adp:
                print("FAIL: AdapterPicker must use studio.loraLabel / studio.noLora (plan D)")
                return 1
            print("OK: AdapterPicker.js (plan D)")
        if IMAGE_EDITOR_JS.exists():
            ie = IMAGE_EDITOR_JS.read_text(encoding="utf-8")
            if "pick-edit-source" not in ie:
                print("FAIL: ImageEditor must declare emit pick-edit-source (plan C4)")
                return 1
            if "editImageEmptyHint" not in ie:
                print(
                    "FAIL: ImageEditor empty state must reference studio.editImageEmptyHint "
                    "(plan C4: AssetPicker lives on ImageCreatePage card header to avoid duplicate UI)"
                )
                return 1
            if "@keydown=\"onKeyDown\"" not in ie:
                print("FAIL: ImageEditor must bind @keydown=onKeyDown on root (plan C6)")
                return 1
            if "onKeyDown," not in ie:
                print("FAIL: ImageEditor setup return must list onKeyDown (plan C6)")
                return 1
            if "studio.brush" not in ie or "studio.clearMask" not in ie:
                print("FAIL: ImageEditor must use studio.* for mask toolbar labels (plan D)")
                return 1
            print("OK: ImageEditor.js (plan C4+C6)")
        if IMAGE_CREATE_PAGE_JS.exists():
            ic2 = IMAGE_CREATE_PAGE_JS.read_text(encoding="utf-8")
            if "<asset-picker" not in ic2:
                print("FAIL: ImageCreatePage must render <asset-picker> for ref / edit sources")
                return 1
            if "control-recent-gallery" not in ic2 or "@control-asset-pick" not in ic2:
                print("FAIL: ImageCreatePage must pass control-recent-gallery and @control-asset-pick to RegistryParamsForm")
                return 1
            if "@pick-edit-source" not in ic2 or ':recent-gallery="recentImages"' not in ic2:
                print("FAIL: ImageCreatePage must wire image-editor pick-edit-source + recent-gallery")
                return 1
            _five_tabs = (
                "action.image.create",
                "action.image.retouch",
                "action.image.extend",
                "action.image.upscale",
            )
            _rewrite_keys = ("action.image.rewrite", "create.rewriteDriveReference", "create.rewriteDriveInstruct")
            if not all(k in ic2 for k in _five_tabs) or not any(k in ic2 for k in _rewrite_keys):
                print(
                    "FAIL: ImageCreatePage must expose all five image action tab keys "
                    "(plan §2.1: create / rewrite / retouch / extend / upscale)"
                )
                return 1
            if "studio.recent" not in ic2 or ("studio.recommendedVersion" not in ic2 and "studio.recommended" not in ic2):
                print("FAIL: ImageCreatePage must use studio.* for shared model/gallery UI (plan D)")
                return 1
            if "studio.generate" not in ic2 or "studio.needControlImage" not in ic2:
                print("FAIL: ImageCreatePage must use studio.generate / studio.needControlImage (plan D)")
                return 1
            if "studio.prompt" not in ic2 or "studio.negativePrompt" not in ic2:
                print("FAIL: ImageCreatePage must use studio.prompt / studio.negativePrompt (plan D)")
                return 1
            if "studio.switchModel" not in ic2:
                print("FAIL: ImageCreatePage must use studio.switchModel when model version changes (plan D)")
                return 1
            if "ensureQueuePoller" in ic2:
                print("FAIL: ImageCreatePage must not call TasksStore.ensureQueuePoller (owned by app root)")
                return 1
            if "DQMemoryHint" not in ic2 or "warnIfRisky" not in ic2:
                print("FAIL: ImageCreatePage must call DQMemoryHint.warnIfRisky before submit (plan E4)")
                return 1
            if "DQTaskStatusUi" not in ic2 or "DQModelVersionValue" not in ic2:
                print("FAIL: ImageCreatePage must use DQTaskStatusUi + DQModelVersionValue composables")
                return 1
            if "DQStudioNav" not in ic2:
                print("FAIL: ImageCreatePage must use DQStudioNav for settings/models navigation")
                return 1
            if "presetActionFilter" not in ic2 or "applies_to" not in ic2 or "media_scope" not in ic2:
                print("FAIL: ImageCreatePage must filter presets via applies_to + media_scope (plan G2)")
                return 1
        if VIDEO_CREATE_PAGE_JS.exists():
            vc = VIDEO_CREATE_PAGE_JS.read_text(encoding="utf-8")
            if "<asset-picker" not in vc:
                print("FAIL: VideoCreatePage must render <asset-picker> for start image")
                return 1
            if "action.video.startImage" not in vc:
                print("FAIL: VideoCreatePage must use action.video.startImage (plan D)")
                return 1
            if "studio.recent" not in vc or ("studio.recommendedVersion" not in vc and "studio.recommended" not in vc):
                print("FAIL: VideoCreatePage must use studio.* for shared UI (plan D)")
                return 1
            if "studio.logs" not in vc:
                print("FAIL: VideoCreatePage must use studio.logs (plan D)")
                return 1
            if "studio.startImageAdded" not in vc:
                print("FAIL: VideoCreatePage must use studio.startImageAdded (plan D)")
                return 1
            if "studio.generate" not in vc and "action.video.create" not in vc:
                print(
                    "FAIL: VideoCreatePage must reference studio.generate or action.video.create (plan §3.1 CTA)"
                )
                return 1
            if "studio.prompt" not in vc or "studio.negativePrompt" not in vc:
                print("FAIL: VideoCreatePage must use studio.prompt / studio.negativePrompt (plan D)")
                return 1
            if "studio.steps" not in vc or "studio.switchModel" not in vc:
                print("FAIL: VideoCreatePage must use studio.steps and log studio.switchModel (plan D)")
                return 1
            if "video.runtimeCardTitle" not in vc or "outputClipSecRounded" not in vc:
                print("FAIL: VideoCreatePage must show plan §3.2 runtime / resource hints (video.runtime*)")
                return 1
            if "ensureQueuePoller" in vc:
                print("FAIL: VideoCreatePage must not call TasksStore.ensureQueuePoller (owned by app root)")
                return 1
            if "DQMemoryHint" not in vc or "warnIfRisky" not in vc:
                print("FAIL: VideoCreatePage must call DQMemoryHint.warnIfRisky before submit (plan E4)")
                return 1
            if "DQTaskStatusUi" not in vc or "DQModelVersionValue" not in vc:
                print("FAIL: VideoCreatePage must use DQTaskStatusUi + DQModelVersionValue composables")
                return 1
            if "DQStudioNav" not in vc:
                print("FAIL: VideoCreatePage must use DQStudioNav for models navigation")
                return 1
            if "presetActionFilter" not in vc or "applies_to" not in vc or "media_scope" not in vc:
                print("FAIL: VideoCreatePage must filter presets via applies_to + media_scope (plan G2)")
                return 1
        if SETTINGS_PAGE_JS.exists():
            sp = SETTINGS_PAGE_JS.read_text(encoding="utf-8")
            if "actionTagLabel" not in sp or "action.image." not in sp:
                print("FAIL: SettingsPage must define actionTagLabel using action.image.* (plan D)")
                return 1
            if "capability-tag" in sp:
                print("FAIL: SettingsPage must not use legacy class capability-tag")
                return 1
            if "studio.recommended" not in sp:
                print("FAIL: SettingsPage must use studio.recommended for model list badge (plan D)")
                return 1
            if "api.gen.getQueue" not in sp or "queuePreviewTitle" not in sp:
                print("FAIL: SettingsPage queue snapshot must use api.gen.getQueue + i18n queuePreviewTitle (Plan E2)")
                return 1
            if "registry-params-form" not in sp or "RegistryParamSchema" not in sp:
                print(
                    "FAIL: SettingsPage must use registry-params-form + RegistryParamSchema for per-model defaults (plan C3)"
                )
                return 1
            if "onSettingsModelRestoreDefaults" not in sp or "settingsLorasForForm" not in sp:
                print("FAIL: SettingsPage must wire LoRA list + restore-defaults for model tab (plan C3)")
                return 1
        print("OK: RegistryParamsForm + schema bundle")

    return 0


if __name__ == "__main__":
    sys.exit(main())
