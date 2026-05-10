"""SeedVR2 超分运行时（Studio 专用）。

编排入口为 ``seedvr2.upscale_pipeline.SeedVR2UpscalePipeline``；本包提供 3D VAE、DiT、
扁平 bundle 权重映射与加载、单步 ``seedvr2_euler`` 调度、空回调注册表、张量→PIL。
命名 ``runtime`` 表示与通用 ``ImagePipeline`` 去噪骨架并行的一层数值实现。

已移除：独立 CLI、LoRA 路径解析、Flux/线性/外部调度、动态分辨率推断、回调管理器
与各类中间件 saver（丹青路径从不注册）。
"""
