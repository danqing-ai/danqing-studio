"""Transformer / 权重映射 / Text Encoder 注册表 — 新增模型只需在此添加条目。"""

# (模块路径, 类名)
_TRANSFORMER = {
    "z_image":  ("backend.engine.z_image.transformer",  "ZImageTransformer"),
    "flux2":    ("backend.engine.flux2.transformer",     "Flux2Transformer"),
    "fibo":     ("backend.engine.fibo.transformer",      "FIBOTransformer"),
    "longcat":  ("backend.engine.longcat.transformer",   "LongCatTransformer"),
    "flux1":    ("backend.engine.flux1.transformer",     "Flux1Transformer"),
}

_WEIGHT_REMAP = {
    "z_image":  ("backend.engine.z_image.weights",  "remap_zimage_weights"),
    "flux2":    ("backend.engine.flux2.weights",     "remap_flux2_weights"),
    "longcat":  ("backend.engine.longcat.weights",   "remap_longcat_weights"),
}

# encoder_type → (模块路径, 类名)
_TEXT_ENCODER = {
    "flux2":    ("backend.engine.flux2.text_encoder",     "Flux2TextEncoder"),
    "z_image":  ("backend.engine.z_image.text_encoder",   "ZImageTextEncoder"),
    "qwen25vl": ("backend.engine.longcat.text_encoder", "LongCatTextEncoder"),
}


def get_transformer_class(family: str):
    import importlib
    entry = _TRANSFORMER.get(family)
    if entry is None:
        raise RuntimeError(f"Unknown image model family: {family}")
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_weight_remap(family: str):
    import importlib
    entry = _WEIGHT_REMAP.get(family)
    if entry is None:
        return None
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_text_encoder(encoder_type: str):
    import importlib
    entry = _TEXT_ENCODER.get(encoder_type)
    if entry is None:
        raise RuntimeError(f"Unknown encoder type: {encoder_type}")
    return getattr(importlib.import_module(entry[0]), entry[1])
