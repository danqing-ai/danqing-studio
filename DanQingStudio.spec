# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['/Users/nil.luo/Workspace/coding/mflux-studio/backend/main.py'],
    pathex=[],
    binaries=[('/Users/nil.luo/Workspace/coding/mflux-studio/.venv/lib/python3.11/site-packages/mlx/lib/libjaccl.dylib', 'mlx/lib'), ('/Users/nil.luo/Workspace/coding/mflux-studio/.venv/lib/python3.11/site-packages/mlx/lib/libmlx.dylib', 'mlx/lib')],
    datas=[('/Users/nil.luo/Workspace/coding/mflux-studio/frontend', 'frontend'), ('/Users/nil.luo/Workspace/coding/mflux-studio/config/locales', 'config/locales'), ('/Users/nil.luo/Workspace/coding/mflux-studio/config/models_registry.json', 'config'), ('/Users/nil.luo/Workspace/coding/mflux-studio/config/presets.json', 'config')],
    hiddenimports=['uvicorn.protocols.http.auto', 'uvicorn.protocols.websockets.auto', 'uvicorn.loops.auto', 'uvicorn.logging', 'fastapi.middleware.cors', 'fastapi.staticfiles', 'backend.api.routes.adapters', 'backend.api.routes.assets', 'backend.api.routes.audios', 'backend.api.routes.download', 'backend.api.routes.gallery', 'backend.api.routes.images', 'backend.api.routes.models', 'backend.api.routes.presets', 'backend.api.routes.queue', 'backend.api.routes.registry', 'backend.api.routes.settings', 'backend.api.routes.system', 'backend.api.routes.tasks', 'backend.api.routes.videos', 'backend.core.container', 'backend.core.i18n', 'backend.core.interfaces', 'backend.core.contracts', 'backend.core.asset_interfaces', 'backend.core.media_interfaces', 'backend.core.model_registry', 'backend.core.registry_format', 'backend.core.task_kinds', 'backend.engine.engine_registry', 'backend.engine.base', 'backend.engine.mlx_runtime', 'backend.engine.model_cache', 'backend.engine.image.mflux_engine', 'backend.engine.image.mflux_generation_backend', 'backend.engine.image.pipeline', 'backend.engine.video.mlx_generation_backend', 'backend.engine.video.mlx_video_engine', 'backend.engine.video.pipeline', 'backend.engine.image.families', 'backend.engine.image.families._base', 'backend.engine.image.families._wired', 'backend.engine.image.families.controlnet', 'backend.engine.image.families.fibo', 'backend.engine.image.families.flux1', 'backend.engine.image.families.flux2', 'backend.engine.image.families.kontext', 'backend.engine.image.families.qwen_image', 'backend.engine.image.families.redux', 'backend.engine.image.families.seedvr2', 'backend.engine.image.families.z_image', 'backend.engine.video.families', 'backend.engine.video.families._base', 'backend.engine.video.families._wired', 'backend.engine.video.families.ltx', 'backend.engine.video.families.wan', 'backend.services.services', 'backend.services.download_service', 'backend.persistence.stores', 'backend.persistence.asset_store', 'backend.persistence.v3_task_store', 'backend.persistence.task_store', 'backend.scheduler.task_scheduler', 'backend.utils.path_utils', 'PIL', 'PIL._imagingtk', 'PIL._tkinter_finder', 'psutil', 'aiohttp', 'python_multipart', 'pydantic', 'huggingface_hub', 'safetensors', 'tqdm', 'requests', 'mlx', 'mlx.core', 'mlx._reprlib_fix', 'mflux'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['/Users/nil.luo/Workspace/coding/mflux-studio/scripts/pyinstaller_runtime_hook.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DanQingStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='arm64',
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DanQingStudio',
)
app = BUNDLE(
    coll,
    name='DanQingStudio.app',
    icon=None,
    bundle_identifier='com.danqing.studio',
)
