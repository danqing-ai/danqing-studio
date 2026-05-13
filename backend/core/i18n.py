"""Backend i18n translation service"""

import json
import os
import sys
from pathlib import Path
from typing import Optional


_translations: dict = {}
_current_locale: str = "zh"


def _load_translations():
    global _translations
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # After PyInstaller packaging
        exe_dir = Path(sys.executable).parent.resolve()
        # macOS .app bundle: executable in Contents/MacOS/, data in Contents/Resources/
        if sys.platform == "darwin" and exe_dir.name == "MacOS" and (exe_dir.parent / "Resources").exists():
            base_dir = exe_dir.parent / "Resources"
        else:
            base_dir = Path(sys._MEIPASS)
        locales_dir = base_dir / "config" / "locales"
    else:
        locales_dir = Path(__file__).parent.parent.parent / "config" / "locales"
    _translations = {}
    if locales_dir.exists():
        for f in locales_dir.iterdir():
            if f.suffix == ".json":
                locale = f.stem
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        _translations[locale] = json.load(fp)
                except Exception:
                    _translations[locale] = {}


def _get_nested(d: dict, key: str, default: str = "") -> str:
    parts = key.split(".")
    for part in parts:
        if isinstance(d, dict):
            d = d.get(part)
            if d is None:
                return default
        else:
            return default
    return str(d) if d is not None else default


def set_locale(locale: str):
    global _current_locale
    _current_locale = locale if locale in _translations else "zh"


def get_locale() -> str:
    return _current_locale


def t(key: str, locale: Optional[str] = None, **params) -> str:
    if not _translations:
        _load_translations()
    loc = locale or _current_locale
    msg = _get_nested(_translations.get(loc, {}), key, key)
    if params:
        try:
            msg = msg.format(**params)
        except KeyError:
            pass
    return msg


def resolve_locale(accept_language: Optional[str]) -> str:
    if not accept_language:
        return "zh"
    for part in accept_language.split(","):
        part = part.strip().split(";")[0].split("-")[0].lower()
        if part in ("zh", "en"):
            return part
    return "zh"
