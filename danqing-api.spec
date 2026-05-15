# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['/Users/nil.luo/Workspace/coding/DanQing-Studio/backend/main.py'],
    pathex=[],
    binaries=[('/Users/nil.luo/Workspace/coding/DanQing-Studio/.venv/lib/python3.11/site-packages/mlx/lib/libjaccl.dylib', 'mlx/lib'), ('/Users/nil.luo/Workspace/coding/DanQing-Studio/.venv/lib/python3.11/site-packages/mlx/lib/libmlx.dylib', 'mlx/lib')],
    datas=[('/Users/nil.luo/Workspace/coding/DanQing-Studio/frontend', 'frontend'), ('/Users/nil.luo/Workspace/coding/DanQing-Studio/config/locales', 'config/locales'), ('/Users/nil.luo/Workspace/coding/DanQing-Studio/config/models_registry.json', 'config'), ('/Users/nil.luo/Workspace/coding/DanQing-Studio/config/presets.json', 'config')],
    hiddenimports=['uvicorn.protocols.http.auto', 'uvicorn.protocols.websockets.auto', 'uvicorn.loops.auto', 'uvicorn.logging', 'fastapi.middleware.cors', 'fastapi.staticfiles', 'backend.api.routes.adapters', 'backend.api.routes.assets', 'backend.api.routes.audios', 'backend.api.routes.download', 'backend.api.routes.gallery', 'backend.api.routes.images', 'backend.api.routes.models', 'backend.api.routes.presets', 'backend.api.routes.queue', 'backend.api.routes.registry', 'backend.api.routes.settings', 'backend.api.routes.system', 'backend.api.routes.tasks', 'backend.api.routes.videos', 'backend.core.container', 'backend.core.i18n', 'backend.core.interfaces', 'backend.core.contracts', 'backend.core.asset_interfaces', 'backend.core.media_interfaces', 'backend.core.model_registry', 'backend.core.registry_format', 'backend.core.task_kinds', 'backend.engine.engine_registry', 'backend.engine.base', 'backend.engine.mlx_runtime', 'backend.engine.model_cache', 'backend.engine.danqing_image_engine', 'backend.engine.danqing_video_engine', 'backend.engine.danqing_audio_engine', 'backend.engine.pipelines', 'backend.engine.pipelines.image_pipeline', 'backend.engine.pipelines.image_upscale_pipeline', 'backend.engine.pipelines.video_pipeline', 'backend.engine.pipelines.video_upscale_pipeline', 'backend.engine.common.safetensors_affine_quant', 'backend.engine._transformer_registry', 'backend.engine.families', 'backend.engine.families.fibo', 'backend.engine.families.flux1', 'backend.engine.families.flux2', 'backend.engine.families.qwen', 'backend.engine.families.z_image', 'backend.engine.families.z_image.text_encoder_cuda', 'backend.engine.families.seedvr2', 'backend.engine.families.ltx', 'backend.engine.families.wan', 'backend.engine.families.cogvideox', 'backend.engine.common.text_encoders.clip_cuda', 'backend.engine.common.text_encoders.t5_cuda', 'backend.engine.common.text_encoders.qwen25vl_cuda', 'backend.services.services', 'backend.services.download_service', 'backend.persistence.stores', 'backend.persistence.asset_store', 'backend.persistence.v3_task_store', 'backend.scheduler.task_scheduler', 'backend.utils.path_utils', 'PIL', 'PIL._imagingtk', 'PIL._tkinter_finder', 'psutil', 'aiohttp', 'python_multipart', 'pydantic', 'huggingface_hub', 'safetensors', 'tqdm', 'requests', 'mlx', 'mlx.core', 'mlx._reprlib_fix'],
    hookspath=['/Users/nil.luo/Workspace/coding/DanQing-Studio/scripts/pyinstaller_hooks'],
    hooksconfig={},
    runtime_hooks=['/Users/nil.luo/Workspace/coding/DanQing-Studio/scripts/pyinstaller_runtime_hook.py'],
    excludes=['tensorboard', 'tensorboard_data_server', 'torch.utils.tensorboard'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='danqing-api',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    name='danqing-api',
)
