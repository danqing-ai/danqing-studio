
import os
import sys
from pathlib import Path

# PyInstaller 运行时：在可执行文件所在目录创建用户数据目录
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # 使用可执行文件所在目录作为用户数据根目录
    app_dir = Path(sys.executable).parent.resolve()
    for dir_name in ['models', 'outputs', 'db']:
        (app_dir / dir_name).mkdir(parents=True, exist_ok=True)
