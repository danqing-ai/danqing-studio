import os
import sys
from pathlib import Path

# PyInstaller: ensure writable dirs exist (Tauri sets DANQING_USER_DATA_DIR).
if getattr(sys, "frozen", False):
    raw = os.environ.get("DANQING_USER_DATA_DIR")
    if raw:
        app_dir = Path(raw).expanduser().resolve()
    else:
        app_dir = Path(sys.executable).parent.resolve()
    for dir_name in ("models", "outputs", "db", "config"):
        (app_dir / dir_name).mkdir(parents=True, exist_ok=True)
