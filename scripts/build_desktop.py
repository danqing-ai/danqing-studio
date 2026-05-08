#!/usr/bin/env python3
"""
DanQing Studio 桌面程序打包脚本
使用 PyInstaller 将后端打包为独立可执行文件或 macOS App

用法:
    python scripts/build_desktop.py           # 构建 standalone 可执行文件
    python scripts/build_desktop.py --app     # 构建 macOS .app bundle

输出:
    dist/DanQingStudio/          (standalone)
    dist/DanQingStudio.app/      (macOS app bundle)
"""

import sys
import os
import argparse
import shutil
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def get_hidden_imports():
    """收集所有需要显式包含的隐藏导入"""
    return [
        # uvicorn 协议和循环
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.loops.auto",
        "uvicorn.logging",
        
        # FastAPI 相关
        "fastapi.middleware.cors",
        "fastapi.staticfiles",
        
        # 后端 API 路由模块
        "backend.api.routes.adapters",
        "backend.api.routes.assets",
        "backend.api.routes.audios",
        "backend.api.routes.download",
        "backend.api.routes.gallery",
        "backend.api.routes.images",
        "backend.api.routes.models",
        "backend.api.routes.presets",
        "backend.api.routes.queue",
        "backend.api.routes.registry",
        "backend.api.routes.settings",
        "backend.api.routes.system",
        "backend.api.routes.tasks",
        "backend.api.routes.videos",
        
        # 后端核心模块
        "backend.core.container",
        "backend.core.i18n",
        "backend.core.interfaces",
        "backend.core.contracts",
        "backend.core.asset_interfaces",
        "backend.core.media_interfaces",
        "backend.core.model_registry",
        "backend.core.registry_format",
        "backend.core.task_kinds",
        
        # 后端引擎模块
        "backend.engine.engine_registry",
        "backend.engine.base",
        "backend.engine.mlx_runtime",
        "backend.engine.model_cache",
        "backend.engine.image.mflux_engine",
        "backend.engine.image.mflux_generation_backend",
        "backend.engine.image.pipeline",
        "backend.engine.video.mlx_generation_backend",
        "backend.engine.video.mlx_video_engine",
        "backend.engine.video.pipeline",
        "backend.engine.image.families",
        "backend.engine.image.families._base",
        "backend.engine.image.families._wired",
        "backend.engine.image.families.controlnet",
        "backend.engine.image.families.fibo",
        "backend.engine.image.families.flux1",
        "backend.engine.image.families.flux2",
        "backend.engine.image.families.kontext",
        "backend.engine.image.families.qwen_image",
        "backend.engine.image.families.redux",
        "backend.engine.image.families.seedvr2",
        "backend.engine.image.families.z_image",
        "backend.engine.video.families",
        "backend.engine.video.families._base",
        "backend.engine.video.families._wired",
        "backend.engine.video.families.ltx",
        "backend.engine.video.families.wan",
        
        # 后端服务模块
        "backend.services.services",
        "backend.services.download_service",
        
        # 后端持久化模块
        "backend.persistence.stores",
        "backend.persistence.asset_store",
        "backend.persistence.v3_task_store",
        "backend.persistence.task_store",
        
        # 后端调度器
        "backend.scheduler.task_scheduler",
        
        # 后端工具
        "backend.utils.path_utils",
        
        # 第三方库
        "PIL",
        "PIL._imagingtk",
        "PIL._tkinter_finder",
        "psutil",
        "aiohttp",
        "python_multipart",
        "pydantic",
        "huggingface_hub",
        "safetensors",
        "tqdm",
        "requests",
        
        # MLX 相关（macOS only）
        "mlx",
        "mlx.core",
        "mlx._reprlib_fix",
        "mflux",
    ]


def get_data_files():
    """收集需要包含的数据文件"""
    data = []
    separator = ";" if sys.platform == "win32" else ":"
    
    # 前端文件
    frontend_dir = PROJECT_ROOT / "frontend"
    if frontend_dir.exists():
        data.append(f"{frontend_dir}{separator}frontend")
    
    # 配置文件
    config_dir = PROJECT_ROOT / "config"
    if config_dir.exists():
        # 包含 locales 目录
        locales_dir = config_dir / "locales"
        if locales_dir.exists():
            data.append(f"{locales_dir}{separator}config/locales")
        
        # 包含 models_registry.json
        registry_file = config_dir / "models_registry.json"
        if registry_file.exists():
            data.append(f"{registry_file}{separator}config")
        
        # 包含 presets.json
        presets_file = config_dir / "presets.json"
        if presets_file.exists():
            data.append(f"{presets_file}{separator}config")
    
    return data


def get_binary_files():
    """收集需要包含的二进制文件（动态库等）"""
    binaries = []
    separator = ";" if sys.platform == "win32" else ":"
    
    # MLX 额外的动态库（PyInstaller 不会自动包含）
    if sys.platform == "darwin":
        # 查找虚拟环境中的 mlx 库
        venv_site_packages = PROJECT_ROOT / ".venv" / "lib" / "python3.11" / "site-packages"
        mlx_lib = venv_site_packages / "mlx" / "lib"
        
        if mlx_lib.exists():
            # 包含所有 .dylib 文件
            for dylib in mlx_lib.glob("*.dylib"):
                # 目标路径保持 mlx/lib/ 结构
                binaries.append(f"{dylib}{separator}mlx/lib")
    
    return binaries


def get_runtime_hooks():
    """运行时钩子"""
    hooks = []
    
    # 创建运行时钩子文件
    hook_file = PROJECT_ROOT / "scripts" / "pyinstaller_runtime_hook.py"
    hook_content = '''
import os
import sys
from pathlib import Path

# PyInstaller 运行时：在可执行文件所在目录创建用户数据目录
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # 使用可执行文件所在目录作为用户数据根目录
    app_dir = Path(sys.executable).parent.resolve()
    for dir_name in ['models', 'outputs', 'db']:
        (app_dir / dir_name).mkdir(parents=True, exist_ok=True)
'''
    
    hook_file.parent.mkdir(parents=True, exist_ok=True)
    with open(hook_file, 'w') as f:
        f.write(hook_content)
    
    hooks.append(str(hook_file))
    return hooks


def create_info_plist(app_name, bundle_id, version="3.0.0"):
    """创建 macOS Info.plist"""
    plist_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>
    <key>CFBundleDisplayName</key>
    <string>{app_name}</string>
    <key>CFBundleExecutable</key>
    <string>{app_name}</string>
    <key>CFBundleIdentifier</key>
    <string>{bundle_id}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>{app_name}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>{version}</string>
    <key>CFBundleVersion</key>
    <string>{version}</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>'''
    return plist_content


def build_app_bundle():
    """构建 macOS .app bundle"""
    build(mode="app")


def build_standalone():
    """构建 standalone 可执行文件"""
    build(mode="standalone")


def build(mode="standalone"):
    """执行打包"""
    try:
        import PyInstaller.__main__
    except ImportError:
        print("错误: 未安装 PyInstaller")
        print("请先运行: pip install pyinstaller")
        sys.exit(1)
    
    is_app = mode == "app"
    
    print("=" * 60)
    if is_app:
        print("DanQing Studio — macOS App Bundle 打包")
    else:
        print("DanQing Studio — Standalone 打包")
    print("=" * 60)
    
    # 检查项目结构
    entry_point = PROJECT_ROOT / "backend" / "main.py"
    if not entry_point.exists():
        print(f"错误: 找不到入口文件 {entry_point}")
        sys.exit(1)
    
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"入口文件: {entry_point}")
    print(f"构建模式: {mode}")
    
    # 构建命令
    app_name = "DanQingStudio"
    cmd = [
        str(entry_point),
        "--name", app_name,
        "--clean",  # 清理临时文件
        "--noconfirm",  # 不确认覆盖
    ]
    
    if is_app:
        # macOS App bundle 模式
        cmd.extend([
            "--windowed",  # 创建 .app bundle
            "--osx-bundle-identifier", "com.danqing.studio",
            "--target-architecture", "arm64",  # Apple Silicon
        ])
        
        # 创建自定义 Info.plist
        plist_content = create_info_plist(app_name, "com.danqing.studio")
        plist_path = PROJECT_ROOT / "scripts" / "Info.plist"
        with open(plist_path, 'w') as f:
            f.write(plist_content)
        cmd.extend(["--osx-bundle-identifier", "com.danqing.studio"])
    else:
        # Standalone 模式
        cmd.extend(["--console"])
    
    # 添加隐藏导入
    for imp in get_hidden_imports():
        cmd.extend(["--hidden-import", imp])
    
    # 添加数据文件
    for data in get_data_files():
        cmd.extend(["--add-data", data])
    
    # 添加二进制文件（动态库等）
    binaries = get_binary_files()
    for binary in binaries:
        cmd.extend(["--add-binary", binary])
    
    # 添加运行时钩子
    for hook in get_runtime_hooks():
        cmd.extend(["--runtime-hook", hook])
    
    # 输出目录
    cmd.extend(["--distpath", str(PROJECT_ROOT / "dist")])
    cmd.extend(["--workpath", str(PROJECT_ROOT / "build")])
    cmd.extend(["--specpath", str(PROJECT_ROOT)])
    
    print(f"\n打包配置:")
    print(f"  输出名称: {app_name}")
    print(f"  模式: {'macOS App Bundle' if is_app else 'Standalone (onedir)'}")
    print(f"  隐藏导入: {len(get_hidden_imports())} 个")
    print(f"  数据文件: {len(get_data_files())} 个")
    print(f"  二进制文件: {len(binaries)} 个")
    print(f"  运行时钩子: {len(get_runtime_hooks())} 个")
    print()
    
    print("开始打包...")
    print("-" * 60)
    
    # 执行打包
    PyInstaller.__main__.run(cmd)
    
    print("-" * 60)
    print("打包完成!")
    print()
    
    # 显示输出信息
    if is_app and sys.platform == "darwin":
        output_dir = PROJECT_ROOT / "dist" / f"{app_name}.app"
        executable = output_dir / "Contents" / "MacOS" / app_name
        
        # 如果 PyInstaller 没有正确设置 Info.plist，手动替换
        plist_dest = output_dir / "Contents" / "Info.plist"
        if plist_dest.exists():
            shutil.copy2(plist_path, plist_dest)
    else:
        output_dir = PROJECT_ROOT / "dist" / app_name
        if sys.platform == "win32":
            executable = output_dir / f"{app_name}.exe"
        else:
            executable = output_dir / app_name
    
    if output_dir.exists():
        # 计算整个目录大小
        total_size = 0
        for f in output_dir.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size
        total_size_mb = total_size / (1024 * 1024)
        
        print(f"输出目录: {output_dir}")
        if executable.exists():
            print(f"可执行文件: {executable}")
        print(f"总大小: {total_size_mb:.1f} MB")
        print()
        print("运行方式:")
        if is_app and sys.platform == "darwin":
            print(f"  双击: {output_dir}")
            print(f"  或命令行: open {output_dir}")
        else:
            print(f"  {executable}")
    else:
        print("警告: 未找到输出文件")
    
    print()
    print("提示:")
    print("  - 首次运行需要加载模型，请耐心等待")
    print("  - 访问 http://localhost:7860 使用应用")
    if is_app:
        print("  - 用户数据保存在 ~/Library/Application Support/DanQingStudio/")
    else:
        print("  - 用户数据保存在可执行文件所在目录")
    print("  - 模型文件保存在 models/ 目录")
    print("  - 输出文件保存在 outputs/ 目录")
    
    if is_app:
        print()
        print("分发说明:")
        print("  - 打包整个 .app 目录即可分发")
        print("  - 可放入 /Applications 目录安装")
        print("  - 首次运行可能需要在 系统偏好设置 > 安全性与隐私 中允许")


def main():
    parser = argparse.ArgumentParser(
        description="DanQing Studio 桌面程序打包脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python scripts/build_desktop.py           # 构建 standalone 可执行文件
    python scripts/build_desktop.py --app     # 构建 macOS .app bundle
        """
    )
    parser.add_argument(
        "--app",
        action="store_true",
        help="构建 macOS .app bundle (仅限 macOS)"
    )
    
    args = parser.parse_args()
    
    if args.app:
        if sys.platform != "darwin":
            print("错误: --app 模式仅限 macOS")
            sys.exit(1)
        build_app_bundle()
    else:
        build_standalone()


if __name__ == "__main__":
    main()
