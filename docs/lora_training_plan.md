# LoRA 训练功能落地方案

> **状态**: 待复核 | **创建日期**: 2026-05-24 | **优先级**: P1
>
> 基于 mlx-examples LoRA 训练经验，在 DanQing Studio v4 中落地本地轻量 LoRA 训练能力。

---

## 1. 执行摘要

### 1.1 现状

DanQing Studio 当前**仅支持 LoRA 推理**（加载/合并权重），不支持训练。现有基础设施已具备：

- ✅ LoRA 权重加载与注入 (`backend/engine/common/weights/__init__.py`)
- ✅ MLX 运行时 LoRA 合并 (`families/*/lora_mlx.py`)
- ✅ 模型注册表 LoRA 分类 (`default_config/models_registry.json`)
- ✅ 前端 LoRA 选择器 (`ImageCreateAdvancedParams.vue`)
- ✅ 全局任务调度器 (`TaskScheduler`)

### 1.2 目标

新增**本地 LoRA 训练**能力，支持用户上传图片、配置参数、后台训练、自动注册使用。

**硬件基线**: M5 Max 128GB (MLX) — 可训练 7B-13B 基础模型的 LoRA。

### 1.3 范围

**In Scope**:
- 图像模型 LoRA 训练（flux1、flux2、z_image、qwen_image 家族）
- MLX 后端优先
- 基础参数可调（rank、lr、steps、target modules）
- 训练任务纳入全局队列
- 训练产物自动注册为可用 LoRA

**Out of Scope (Phase 1)**:
- CUDA 后端训练
- 视频/音频模型 LoRA 训练
- DreamBooth / Textual Inversion
- 多 GPU 分布式训练
- 高级数据增强

---

## 2. 产品定义

### 2.1 用户场景

| 场景 | 描述 | 典型数据量 |
|------|------|-----------|
| 风格迁移 | 训练特定艺术风格 | 20-50 张参考图 |
| 人物一致性 | 训练特定角色/人物 | 10-30 张角色图 |
| 概念学习 | 训练特定物体/场景 | 15-40 张概念图 |

### 2.2 核心流程

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  1. 上传训练图片  │───→│  2. 配置训练参数  │───→│  3. 提交训练任务  │
│   (Gallery/本地) │    │  (模型/触发词/参数)│    │   (全局队列排队) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  6. 使用 LoRA   │←───│  5. 自动注册    │←───│  4. 后台训练    │
│   (生成时选择)   │    │  (加入适配器列表) │    │  (实时进度/日志) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2.3 训练参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `rank` | 16 | 4-128 | LoRA 低秩维度 |
| `learning_rate` | 1e-4 | 1e-5 - 1e-3 | 学习率 |
| `num_train_steps` | 1000 | 100-10000 | 总训练步数 |
| `save_every_n_steps` | 500 | 100-2000 | 保存 checkpoint 间隔 |
| `target_modules` | `["attn"]` | 自定义 | 注入 LoRA 的模块 |
| `trigger_word` | "" | 任意字符串 | 触发词（嵌入 prompt）|
| `batch_size` | 1 | 1-4 | 批次大小（受内存限制）|
| `resolution` | 1024 | 512-2048 | 训练图片分辨率 |

---

## 3. 用户交互设计

### 3.1 CLI 接口

**新增命令**: `bin/danqing-lora-train`

```bash
# 基础用法
danqing-lora-train \
  --base-model flux1-dev \
  --name "my-style-lora" \
  --trigger-word "mystyle" \
  --images-dir ./my_images/ \
  --steps 1000 \
  --rank 16 \
  --learning-rate 1e-4 \

# 完整参数
danqing-lora-train \
  --base-model flux1-dev:fp16 \
  --name "my-style-lora" \
  --description "自定义艺术风格" \
  --trigger-word "mystyle" \
  --images-dir ./my_images/ \
  --steps 2000 \
  --rank 32 \
  --learning-rate 5e-5 \
  --save-every 500 \
  --target-modules attn \
  --resolution 1024 \
  --output-name "mystyle-v1"
```

### 3.2 REST API

**新增路由**: `backend/api/routes/lora_training.py`

```
POST /api/lora/train          # 提交训练任务
GET  /api/lora/train/status   # 查询训练任务状态（复用 /api/tasks/{id}）
GET  /api/lora/train/logs     # 查询训练日志（复用 /api/tasks/{id}/logs）
DELETE /api/lora/train/{id}   # 取消训练任务（复用 /api/tasks/{id}）
```

**请求体**:
```json
{
  "base_model": "flux1-dev:fp16",
  "name": "my-style-lora",
  "description": "自定义艺术风格LoRA",
  "trigger_word": "mystyle",
  "image_asset_ids": ["ast_xxx", "ast_yyy", "ast_zzz"],
  "rank": 16,
  "learning_rate": 0.0001,
  "num_train_steps": 1000,
  "save_every_n_steps": 500,
  "target_modules": ["attn"],
  "resolution": 1024,
  "priority": "normal"
}
```

**响应**:
```json
{
  "task": {
    "id": "tsk_abc123",
    "kind": "lora.training",
    "status": "queued",
    "queue_position": 3,
    "links": {
      "self": "/api/tasks/tsk_abc123",
      "stream": "/api/tasks/tsk_abc123/stream",
      "cancel": "/api/tasks/tsk_abc123"
    }
  }
}
```

### 3.3 前端 UI

**方案 A（推荐）**: 在 `ModelsView.vue` 新增 **"LoRA 训练"** Tab

**方案 B**: 在 `ImageCreateView.vue` 侧边栏新增训练快捷入口

**推荐方案 A**，理由：
- 与模型管理页面逻辑一致
- 训练产物（LoRA）天然归属模型库
- 不干扰现有创作流程

**界面元素**:

```
┌─────────────────────────────────────────────────────────────┐
│  LoRA 训练                                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  基础模型 *                                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ [▼ flux1-dev (FP16)                                ] │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  训练图片 *                                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ [从图库选择...] 已选择 25 张图片                       │  │
│  │ [📷][🖼️][🎨][📸][🌄]...                              │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  训练名称 *        触发词                                   │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ my-style-lora    │  │ mystyle          │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                             │
│  [▼ 高级参数]                                               │
│    ┌─────────────────────────────────────────────────────┐  │
│    │ Rank:        [━━━●━━━━] 16                          │  │
│    │ 学习率:      [━━●━━━━━] 1e-4                        │  │
│    │ 训练步数:    [━━━●━━━━] 1000                        │  │
│    │ 保存间隔:    [━━●━━━━━] 500                         │  │
│    │ 分辨率:      [▼ 1024x1024                         ] │  │
│    │ Target:      [▼ attention                         ] │  │
│    └─────────────────────────────────────────────────────┘  │
│                                                             │
│  [开始训练]                                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**训练进度监控**:
- 复用全局任务队列 (`useTasksStore` + SSE)
- 实时显示 loss 曲线（可选 Phase 2）
- 训练完成后自动刷新适配器列表

---

## 4. 架构实现

### 4.1 新增任务类型

**文件**: `backend/core/task_kinds.py`

```python
LORA_TRAINING = "lora.training"

ALL_KINDS = frozenset({
    IMAGE_GENERATION,
    IMAGE_EDIT,
    IMAGE_UPSCALE,
    VIDEO_GENERATION,
    VIDEO_EDIT,
    AUDIO_GENERATION,
    AUDIO_EDIT,
    LORA_TRAINING,  # 新增
})

# Registry actions → task kind 映射（训练不通过 registry action 触发）
# 直接通过 API/CLI 提交任务
```

### 4.2 新增 Contract

**文件**: `backend/core/contracts.py`

```python
class LoRATrainingRequest(BaseModel):
    """LoRA 训练请求体 —— 对齐 REST / CLI / Engine"""
    
    base_model: str = Field(..., description="基础模型 ID，如 flux1-dev:fp16")
    name: str = Field(..., min_length=1, max_length=64, description="LoRA 名称")
    description: str = Field(default="", max_length=256)
    trigger_word: str = Field(default="", max_length=64, description="触发词")
    
    # 数据源
    image_asset_ids: list[str] = Field(
        default_factory=list,
        description="训练图片 asset_id 列表（Gallery/Assets 系统）"
    )
    
    # 训练参数
    rank: int = Field(default=16, ge=4, le=128, description="LoRA 低秩维度")
    learning_rate: float = Field(default=1e-4, gt=0, le=1e-2)
    num_train_steps: int = Field(default=1000, ge=100, le=10000)
    save_every_n_steps: int = Field(default=500, ge=100, le=2000)
    target_modules: list[str] = Field(
        default_factory=lambda: ["attn"],
        description="注入 LoRA 的模块列表"
    )
    
    # 数据预处理
    resolution: int = Field(default=1024, ge=512, le=2048)
    center_crop: bool = True
    random_flip: bool = True
    
    # 任务控制
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### 4.3 引擎接口扩展

**文件**: `backend/core/media_interfaces.py`

```python
class IImageEngine(ABC):
    # ... 现有方法 ...
    
    @abstractmethod
    async def train_lora(
        self, request: LoRATrainingRequest, ctx: ExecutionContext
    ) -> EngineResult:
        """训练 LoRA 适配器。"""
        pass
```

### 4.4 引擎实现

**文件**: `backend/engine/danqing_image_engine.py`

```python
async def train_lora(
    self, request: LoRATrainingRequest, ctx: ExecutionContext
) -> EngineResult:
    runtime = self._resolve_runtime(request.base_model)
    pipeline = LoRATrainingPipeline(
        runtime,
        self._registry,
        ctx.asset_store,
        model_cache=self._cache,
        project_root=self._paths.get_project_root(),
    )
    
    def on_progress(p, s, t, msg=None):
        from backend.core.contracts import ProgressEvent
        ctx.on_progress(ProgressEvent(progress=p, step=s, total=t, message=msg))
    
    def on_log(lvl, msg):
        from backend.core.contracts import LogEvent
        ctx.on_log(LogEvent(level=lvl, message=msg))
    
    result = await asyncio.to_thread(
        pipeline.run, request, ctx, on_progress=on_progress, on_log=on_log,
    )
    return result
```

### 4.5 训练流水线

**文件**: `backend/engine/pipelines/lora_training_pipeline.py`

```python
class LoRATrainingPipeline:
    """LoRA 训练流水线 —— 复用现有基础设施。"""
    
    def __init__(self, ctx, registry, asset_store, model_cache=None, project_root=None):
        self.ctx = ctx
        self.registry = registry
        self.asset_store = asset_store
        self.model_cache = model_cache
        self.project_root = project_root
    
    def run(self, request: LoRATrainingRequest, ctx: ExecutionContext, 
            on_progress=None, on_log=None) -> EngineResult:
        
        # ── Phase 1: 准备 ─────────────────────────────
        base_model_id, version = parse_model_version(request.base_model)
        family = self._resolve_family(base_model_id)
        
        if on_log:
            on_log("info", f"Starting LoRA training for {base_model_id} (family={family})")
        
        # ── Phase 2: 加载基础模型 ───────────────────────
        # 复用 ImagePipeline._load_model()
        config = self._get_model_config(family, base_model_id)
        model = self._load_model(family, config, base_model_id, version)
        
        # 冻结基础模型参数
        self._freeze_base_model(model)
        
        # ── Phase 3: 注入 LoRA 层 ──────────────────────
        # 复用 families/<family>/lora_mlx.py 的 key 映射逻辑
        # 但改为训练模式：保留 A/B 矩阵为可训练参数
        lora_layers = self._inject_trainable_lora(
            model, family, rank=request.rank, 
            target_modules=request.target_modules
        )
        
        if on_log:
            on_log("info", f"Injected {len(lora_layers)} LoRA layers (rank={request.rank})")
        
        # ── Phase 4: 加载训练数据 ───────────────────────
        images = self._load_training_images(request.image_asset_ids)
        captions = [request.trigger_word] * len(images)  # 简化：仅使用触发词
        
        # 预处理
        processed_images = self._preprocess_images(
            images, resolution=request.resolution,
            center_crop=request.center_crop,
            random_flip=request.random_flip
        )
        
        # ── Phase 5: 文本编码 ──────────────────────────
        # 复用现有 text encoder
        text_encoder = self._load_text_encoder(family, config)
        text_embeds = text_encoder.encode(captions)
        
        # ── Phase 6: VAE 编码 ──────────────────────────
        # 将图片编码为 latent
        vae = self._load_vae(family, config)
        latents = vae.encode(processed_images)
        
        # ── Phase 7: 训练循环 ──────────────────────────
        optimizer = self._create_optimizer(lora_layers, lr=request.learning_rate)
        
        for step in range(request.num_train_steps):
            if ctx.cancel_token.is_cancelled():
                if on_log:
                    on_log("warning", "Training cancelled by user")
                raise asyncio.CancelledError()
            
            # 随机采样 batch
            idx = step % len(latents)
            batch_latents = latents[idx:idx+1]
            batch_embeds = text_embeds[idx:idx+1]
            
            # Flow matching 训练
            loss = self._training_step(
                model, batch_latents, batch_embeds,
                family, config
            )
            
            # 反向传播
            grad_fn = mx.grad(loss)
            grads = grad_fn(*[p for _, p in lora_layers])
            optimizer.update([p for _, p in lora_layers], grads)
            mx.eval([p for _, p in lora_layers])
            
            # 进度报告
            if on_progress and step % 10 == 0:
                progress = step / request.num_train_steps
                on_progress(progress, step, request.num_train_steps, 
                           f"loss={float(loss):.4f}")
            
            # 保存 checkpoint
            if step > 0 and step % request.save_every_n_steps == 0:
                self._save_checkpoint(model, request, step)
        
        # ── Phase 8: 保存最终权重 ──────────────────────
        output_path = self._save_lora_weights(model, request)
        
        if on_log:
            on_log("success", f"LoRA training completed: {output_path}")
        
        # ── Phase 9: 注册到系统 ────────────────────────
        lora_asset_id = self._register_lora(output_path, request)
        
        return EngineResult(
            primary_asset_id=lora_asset_id,
            metadata={
                "type": "lora_training",
                "base_model": base_model_id,
                "rank": request.rank,
                "steps": request.num_train_steps,
                "output_path": str(output_path),
            }
        )
    
    def _training_step(self, model, latents, text_embeds, family, config):
        """单步训练 —— Flow Matching / SD3 损失。"""
        import mlx.core as mx
        
        # 随机 timestep
        t = mx.random.uniform(shape=(1,))
        
        # 加噪
        noise = mx.random.normal(latents.shape)
        noisy_latents = (1 - t) * latents + t * noise
        
        # 模型预测
        pred = model(noisy_latents, t, txt_embeds=text_embeds)
        
        # Flow matching 损失: 预测速度场
        target = noise - latents
        loss = mx.mean((pred - target) ** 2)
        
        return loss
```

### 4.6 家族特定实现

每个支持训练的家族需实现：

**文件**: `backend/engine/families/<family>/lora_train_mlx.py`

```python
"""Family-specific LoRA training utilities."""

def inject_trainable_lora_layers(model, rank: int, target_modules: list[str]):
    """
    在模型中注入可训练的 LoRA 层。
    
    复用 lora_mlx.py 中的 key 映射逻辑，但：
    1. 不合并权重
    2. 保留 A/B 矩阵为 mx.array（可训练）
    3. 返回 (name, parameter) 列表供 optimizer 使用
    """
    lora_params = []
    
    for module_name in target_modules:
        # 查找目标线性层
        linear_layer = find_linear_layer(model, module_name)
        if linear_layer is None:
            continue
        
        in_dim = linear_layer.weight.shape[1]
        out_dim = linear_layer.weight.shape[0]
        
        # 创建 LoRA A/B 矩阵
        lora_A = mx.random.normal((rank, in_dim)) * 0.01
        lora_B = mx.zeros((out_dim, rank))
        
        # 附加到层
        linear_layer.lora_A = lora_A
        linear_layer.lora_B = lora_B
        
        # 修改 forward 以应用 LoRA
        original_forward = linear_layer.forward
        
        def make_lora_forward(orig, A, B, alpha, rank):
            def forward(x):
                base = orig(x)
                delta = mx.matmul(x, A.T)  # [B, rank]
                delta = mx.matmul(delta, B.T)  # [B, out_dim]
                return base + (alpha / rank) * delta
            return forward
        
        linear_layer.forward = make_lora_forward(
            original_forward, lora_A, lora_B, rank, rank
        )
        
        lora_params.extend([
            (f"{module_name}.lora_A", lora_A),
            (f"{module_name}.lora_B", lora_B),
        ])
    
    return lora_params

def extract_lora_weights(model) -> dict:
    """从模型中提取训练后的 LoRA 权重（仅 A/B 矩阵）。"""
    weights = {}
    for name, param in model.parameters():
        if "lora_A" in name or "lora_B" in name:
            weights[name] = param
    return weights
```

### 4.7 任务调度器集成

**文件**: `backend/scheduler/task_scheduler.py`

```python
async def _execute(self, tid: str) -> None:
    # ... 现有代码 ...
    
    if kind == TK.IMAGE_GENERATION:
        req = ImageGenerationRequest.model_validate(params)
        res = await self._engines.get_image(model_id).generate(req, ctx)
    elif kind == TK.IMAGE_EDIT:
        req = ImageEditRequest.model_validate(params)
        res = await self._engines.get_image(model_id).edit(req, ctx)
    # ... 其他任务类型 ...
    elif kind == TK.LORA_TRAINING:
        req = LoRATrainingRequest.model_validate(params)
        res = await self._engines.get_image(model_id).train_lora(req, ctx)
    else:
        raise RuntimeError(f"unknown kind {kind}")
    
    # ... 后续处理 ...
```

### 4.8 模型注册表更新

**文件**: `default_config/models_registry.json`（基础模型参数扩展）

```json
{
  "flux1-dev": {
    "category": "base_models",
    "engine": "danqing-image",
    "parameters": {
      "lora_support": true,
      "lora_training": {
        "supported": true,
        "default_rank": 16,
        "max_rank": 128,
        "target_modules": {
          "options": ["attn", "ff", "all"],
          "default": "attn"
        },
        "default_steps": 1000,
        "max_steps": 10000
      }
    }
  }
}
```

**运行时注册**（训练完成后动态添加）：

```python
def _register_lora(self, output_path: Path, request: LoRATrainingRequest) -> str:
    """将训练好的 LoRA 注册到系统。"""
    
    # 1. 创建 LoRA 配置
    lora_config = {
        "lora_alpha": request.rank,
        "lora_rank": request.rank,
        "base_model": request.base_model,
        "trigger_word": request.trigger_word,
    }
    
    config_path = output_path.with_suffix(".json")
    with open(config_path, "w") as f:
        json.dump(lora_config, f, indent=2)
    
    # 2. 注册到内存中的模型注册表
    registry_entry = {
        "id": request.name,
        "category": "loras",
        "engine": "danqing-image",
        "type": "lora",
        "base_model": request.base_model.split(":")[0],
        "parameters": {
            "lora_scale": {
                "default": 0.8,
                "min": 0.0,
                "max": 2.0,
                "step": 0.1,
                "type": "float"
            }
        },
        "versions": {
            "fp16": {
                "local_path": str(output_path.parent),
                "source_type": "user_trained"
            }
        },
        "name": {"zh": request.name, "en": request.name},
        "description": {"zh": request.description, "en": request.description},
        "media": "image",
        "actions": {},
        "family": self._resolve_family(request.base_model),
    }
    
    # 3. 添加到注册表
    self.registry.register_custom_lora(request.name, registry_entry)
    
    # 4. 持久化到用户配置
    self._persist_user_lora(request.name, registry_entry)
    
    # 5. 创建 asset
    asset_id = new_asset_id()
    self.asset_store.create_from_file(
        asset_id=asset_id,
        file_path=output_path,
        mime_type="application/octet-stream",
        metadata={
            "type": "lora_weights",
            "base_model": request.base_model,
            "trigger_word": request.trigger_word,
        }
    )
    
    return asset_id
```

---

## 5. 实施路线图

### Phase 1: MVP（预计 2-3 周）

**Week 1: 基础设施**
- [ ] 新增 `LORA_TRAINING` 任务类型 (`task_kinds.py`)
- [ ] 新增 `LoRATrainingRequest` contract (`contracts.py`)
- [ ] 扩展 `IImageEngine` 接口 (`media_interfaces.py`)
- [ ] 实现 `DanQingImageEngine.train_lora()` (`danqing_image_engine.py`)
- [ ] 任务调度器集成 (`task_scheduler.py`)

**Week 2: 训练流水线**
- [ ] 新建 `LoRATrainingPipeline` 框架
- [ ] 实现 flux1 家族训练逻辑 (`families/flux1/lora_train_mlx.py`)
- [ ] 数据加载与预处理
- [ ] 训练循环（MLX optimizer + grad）
- [ ] 权重保存与加载

**Week 3: API / CLI / 注册**
- [ ] REST API 路由 (`api/routes/lora_training.py`)
- [ ] CLI 命令 (`cli/lora_training_cli.py` + `bin/danqing-lora-train`)
- [ ] LoRA 自动注册逻辑
- [ ] 基础前端页面（模型选择、图片上传、参数表单）

### Phase 2: 完善（预计 1-2 周）

- [ ] 支持多家族（flux2、z_image、qwen_image）
- [ ] 训练过程可视化（loss 曲线、样本预览）
- [ ] 训练参数预设（风格/人物/概念模板）
- [ ] 训练产物自动出现在适配器列表
- [ ] 训练任务取消/恢复支持
- [ ] 训练数据增强（random crop、color jitter）

### Phase 3: 高级功能（可选，后续迭代）

- [ ] 支持多分辨率训练（bucketing）
- [ ] 支持正则化图片（防止过拟合）
- [ ] 支持 DreamBooth / Textual Inversion
- [ ] 支持训练多个 concept
- [ ] 训练结果评估（自动对比图生成）

---

## 6. 技术风险与应对

| 风险 | 影响 | 可能性 | 应对措施 |
|------|------|--------|----------|
| MLX 训练内存不足（>128GB） | 高 | 中 | 仅支持 int8/int4 基础模型；实现 gradient checkpointing；限制 batch_size=1 |
| 训练时间过长阻塞队列 | 高 | 高 | 训练任务使用独立低优先级 band；支持后台进程模式 |
| LoRA 权重格式不兼容 | 中 | 中 | 严格遵循 diffusers/ComfyUI key 命名；导出时验证 |
| 训练图片预处理差异 | 中 | 中 | 复用现有 VAE 预处理逻辑；文档化预处理流程 |
| 注册表动态更新冲突 | 低 | 低 | 运行时内存注册表 + WAL 持久化；避免并发写入 |
| 训练数据隐私 | 中 | 低 | 本地训练不上传云端；加密存储训练图片 |
| 训练结果质量不可控 | 中 | 高 | 提供默认参数预设；训练前预览配置；训练后自动评估 |

---

## 7. 关键代码改动清单

### 7.1 新建文件

| 文件路径 | 说明 | 优先级 |
|----------|------|--------|
| `backend/core/task_kinds.py` | 新增 `LORA_TRAINING` | P0 |
| `backend/core/contracts.py` | 新增 `LoRATrainingRequest` | P0 |
| `backend/core/media_interfaces.py` | 扩展 `IImageEngine` | P0 |
| `backend/engine/pipelines/lora_training_pipeline.py` | **训练流水线核心** | P0 |
| `backend/engine/families/flux1/lora_train_mlx.py` | flux1 家族训练逻辑 | P0 |
| `backend/api/routes/lora_training.py` | REST API | P1 |
| `backend/cli/lora_training_cli.py` | CLI 实现 | P1 |
| `bin/danqing-lora-train` | CLI 入口脚本 | P1 |
| `frontend/src/views/LoRATrainingView.vue` | 前端训练页面 | P1 |

### 7.2 修改文件

| 文件路径 | 改动内容 | 优先级 |
|----------|----------|--------|
| `backend/engine/danqing_image_engine.py` | 实现 `train_lora()` | P0 |
| `backend/scheduler/task_scheduler.py` | `_execute()` 新增分支 | P0 |
| `backend/engine/common/weights/__init__.py` | 新增 LoRA 权重导出 | P0 |
| `frontend/src/router/index.ts` | 新增路由 | P1 |
| `frontend/src/stores/registry.ts` | 支持动态 LoRA 注册 | P1 |
| `default_config/models_registry.json` | 基础模型参数扩展 | P2 |

### 7.3 测试计划

| 测试项 | 说明 | 优先级 |
|--------|------|--------|
| 单元测试 | `LoRATrainingPipeline` 核心逻辑 | P0 |
| 集成测试 | 端到端训练流程（小数据量） | P0 |
| CLI 测试 | `danqing-lora-train` 命令 | P1 |
| API 测试 | `POST /api/lora/train` | P1 |
| 前端测试 | 训练表单提交与进度监控 | P1 |
| 兼容性测试 | 训练产物可被生成流程使用 | P0 |
| 性能测试 | 训练内存/时间基准 | P2 |

---

## 8. 待决策事项

### 8.1 必须决策

1. **训练后端策略**
   - [ ] **A**: 仅支持 MLX（MVP 快速落地，后续扩展 CUDA）
   - [ ] **B**: MLX + CUDA 同时支持（增加复杂度，覆盖更广）
   - **建议**: A（M5 Max 是目标设备，MLX 优化成熟）

2. **训练数据存储**
   - [ ] **A**: 使用现有 Asset/Gallery 系统（引用 asset_id）
   - [ ] **B**: 独立训练数据集管理（新建数据集概念）
   - **建议**: A（复用基础设施，MVP 快速落地）

3. **训练产物注册方式**
   - [ ] **A**: 自动注册（训练完成后立即可用）
   - [ ] **B**: 手动审核后注册（用户确认后添加）
   - **建议**: A（减少摩擦，用户可随时删除）

4. **训练任务队列策略**
   - [ ] **A**: 纳入全局队列（与生成任务混排）
   - [ ] **B**: 独立训练队列（不影响生成任务）
   - **建议**: B（训练任务耗时较长，独立队列更合理）

### 8.2 可选决策

5. **是否支持训练过程中预览**
   - [ ] **A**: 每 N 步生成预览图（消耗额外资源）
   - [ ] **B**: 仅训练完成后生成（Phase 2 再考虑）

6. **是否支持 LoRA 合并**
   - [ ] **A**: 训练时支持合并多个现有 LoRA（复杂度高）
   - [ ] **B**: 仅训练新 LoRA（保持简单）

7. **训练数据格式**
   - [ ] **A**: 仅支持图片 + 触发词（简单）
   - [ ] **B**: 支持图片 + 自定义 caption（需要文本标注 UI）

---

## 9. 参考资源

- **mlx-examples LoRA**: https://github.com/ml-explore/mlx-examples/tree/main/lora
- **Diffusers LoRA 训练**: https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
- **Kohya_ss 训练器**: https://github.com/bmaltais/kohya_ss
- **DanQing Studio 架构**: `docs/dual_platform_architecture.md`
- **新模型集成 checklist**: `docs/engine_new_model_checklist.md`

---

## 10. 附录

### 10.1 MLX LoRA 训练伪代码

```python
import mlx.core as mx
import mlx.optimizers as optim

def train_lora(model, train_data, config):
    # 冻结基础模型
    mx.eval(model.parameters())
    
    # 创建 LoRA 参数（仅 A/B 矩阵可训练）
    lora_params = inject_lora_layers(model, rank=config.rank)
    
    # 优化器
    optimizer = optim.Adam(learning_rate=config.lr)
    
    for step in range(config.steps):
        # 采样
        latents, text_embeds = sample_batch(train_data)
        
        # 随机 timestep
        t = mx.random.uniform(shape=(1,))
        
        # 加噪
        noise = mx.random.normal(latents.shape)
        noisy = (1 - t) * latents + t * noise
        
        # 前向
        pred = model(noisy, t, text_embeds)
        
        # 损失
        target = noise - latents  # flow matching
        loss = mx.mean((pred - target) ** 2)
        
        # 反向
        grad_fn = mx.grad(loss, lora_params)
        grads = grad_fn()
        
        # 更新
        optimizer.update(lora_params, grads)
        mx.eval(lora_params)
        
        if step % 100 == 0:
            print(f"Step {step}: loss={float(loss):.4f}")
    
    # 保存
    save_lora_weights(model, "output.safetensors")
```

### 10.2 内存估算

| 基础模型 | 量化 | 模型内存 | LoRA 内存 (rank=16) | 训练缓冲 | 总估算 |
|----------|------|----------|---------------------|----------|--------|
| Flux1-dev | FP16 | ~24GB | ~200MB | ~8GB | ~32GB |
| Flux1-dev | INT8 | ~12GB | ~200MB | ~6GB | ~18GB |
| Flux2-Klein-9B | FP16 | ~18GB | ~300MB | ~6GB | ~24GB |
| Z-Image | FP16 | ~12GB | ~150MB | ~4GB | ~16GB |

> **注**: M5 Max 128GB 可轻松支持 FP16 基础模型训练，并有充足余量。

---

*本文档由 DanQing Studio 开发团队维护。复核通过后进入实施阶段。*
