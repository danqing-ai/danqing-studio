"""Register family plugins at process startup."""

from __future__ import annotations

_bootstrapped = False


def bootstrap_family_plugins() -> None:
    global _bootstrapped
    if _bootstrapped:
        return
    from backend.engine.families.ace_step.plugin import register_ace_step_plugin
    from backend.engine.families.diffrhythm.plugin import register_diffrhythm_plugin
    from backend.engine.families.cogview4.plugin import register_cogview4_plugin
    from backend.engine.families.ernie_image.plugin import register_ernie_image_plugin
    from backend.engine.families.fibo.plugin import register_fibo_plugin
    from backend.engine.families.flux1.plugin import register_flux1_plugin
    from backend.engine.families.flux2.plugin import register_flux2_plugin
    from backend.engine.families.hunyuan.plugin import register_hunyuan_plugin
    from backend.engine.families.ltx.plugin import register_ltx_plugin
    from backend.engine.families.qwen.plugin import register_qwen_image_plugin
    from backend.engine.families.bernini.plugin import register_bernini_plugin
    from backend.engine.families.real_esrgan.plugin import register_real_esrgan_plugin
    from backend.engine.families.seedvr2.plugin import register_seedvr2_plugin
    from backend.engine.families.wan.plugin import register_wan_plugin
    from backend.engine.families.z_image.plugin import register_z_image_plugin

    register_flux2_plugin()
    register_z_image_plugin()
    register_qwen_image_plugin()
    register_flux1_plugin()
    register_fibo_plugin()
    register_ernie_image_plugin()
    register_cogview4_plugin()
    register_wan_plugin()
    register_ltx_plugin()
    register_hunyuan_plugin()
    register_seedvr2_plugin()
    register_real_esrgan_plugin()
    register_bernini_plugin()
    register_ace_step_plugin()
    register_diffrhythm_plugin()
    _bootstrapped = True
