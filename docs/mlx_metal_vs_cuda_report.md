# MLX Metal vs CUDA 后端行为差异对比报告

## 1. 设备检测 API

### 1.1 后端特定检测

| API | Metal | CUDA | 说明 |
|-----|-------|------|------|
| `mx.metal.is_available()` | ✅ | ❌ | 仅检测Metal后端 |
| `mx.cuda.is_available()` | ❌ | ✅ | 仅检测CUDA后端 |

**关键发现**：
- MLX 没有提供统一的 `mx.is_available()` 或 `mx.gpu.is_available()` 方法
- 必须分别调用 `mx.metal.is_available()` 和 `mx.cuda.is_available()`
- 在 **Linux + CUDA 环境**下，`import mlx.core` 后 `mx.metal.is_available()` 返回 **False**
- `mx.cuda.is_available()` 仅在安装了 `mlx[cuda]` 或 `mlx-cuda` 额外包时可用

### 1.2 项目中的实际做法

```python
# backend/engine/platform.py 的检测逻辑
if sys.platform == "darwin" and platform.machine() == "arm64":
    try:
        importlib.import_module("mlx.core")
        backends.append("mlx")
    except ImportError:
        pass

try:
    torch = importlib.import_module("torch")
    if torch.cuda.is_available():
        backends.append("cuda")
except ImportError:
    pass
```

**注意**：项目实际使用 `torch.cuda.is_available()` 检测CUDA，而非 `mx.cuda.is_available()`，因为CUDA路径使用PyTorch实现。

### 1.3 差异点（需要 if/else 分支）

```python
# 需要分别检测
if mx.metal.is_available():
    backend = "metal"
elif mx.cuda.is_available():
    backend = "cuda"
else:
    backend = "cpu"
```

---

## 2. 内存管理差异

### 2.1 内存限制 API

| API/特性 | Metal | CUDA | 说明 |
|----------|-------|------|------|
| `mx.set_memory_limit()` | ✅ | ⚠️ | 文档明确说明 **"When metal is available..."**，暗示主要设计给Metal |
| `MLX_METAL_MEMORY_LIMIT` 环境变量 | ✅ | ❌ | **Metal专用**，CUDA无效 |
| `mx.get_active_memory()` | ✅ | ✅ | 通用API，但实现可能不同 |
| `mx.get_cache_memory()` | ✅ | ✅ | 通用API |
| `mx.get_peak_memory()` | ✅ | ✅ | 通用API |
| `mx.clear_cache()` | ✅ | ✅ | 通用API |
| `mx.set_cache_limit()` | ✅ | ? | 文档未明确区分后端 |
| `mx.set_wired_limit()` | ✅ | ? | Metal特定概念（wired memory） |

### 2.2 关键差异分析

**`mx.set_memory_limit()` 行为差异**：

根据官方文档：
> "When metal is available the memory limit defaults to 1.5 times the maximum recommended working set size reported by the device."

这意味着：
1. 在Metal上，`set_memory_limit()` 有默认值（1.5x推荐工作集）
2. 在CUDA上，**可能**没有默认值或行为不同
3. 该API是"guideline"（指导值），非硬性限制

**`MLX_METAL_MEMORY_LIMIT` 环境变量**：
- 仅在Metal后端有效
- 项目代码中同时使用环境变量 + API调用：

```python
# backend/engine/runtime/mlx.py
os.environ["MLX_METAL_MEMORY_LIMIT"] = str(self._memory_limit_gb)
try:
    mx.set_memory_limit(self._memory_limit_gb * 1024**3)
except Exception:
    pass
```

### 2.3 内存统计跨后端一致性

```python
# Metal 路径
mx.get_active_memory()   # 返回当前活跃内存（bytes）
mx.get_cache_memory()    # 返回缓存内存（bytes）
mx.get_peak_memory()     # 返回峰值内存（bytes）

# 项目中的封装
# MLXContext.active_memory_gb() -> mx.get_active_memory() / 1024**3
# CudaContext.active_memory_gb() -> torch.cuda.memory_allocated() / 1024**3
```

**注意**：MLX的内存统计API在CUDA上**理论可用**，但实际项目中CudaContext直接使用 `torch.cuda.memory_allocated()`，说明：
- MLX的内存API在CUDA上可能不够精确
- 或与PyTorch的内存管理存在冲突

### 2.4 需要 if/else 分支的场景

```python
# 设置内存限制（Metal专用）
if mx.metal.is_available():
    os.environ["MLX_METAL_MEMORY_LIMIT"] = str(limit_gb)
    mx.set_memory_limit(limit_gb * 1024**3)

# CUDA需要torch管理
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(fraction)
```

---

## 3. Stream 和设备

### 3.1 Stream API 跨后端一致性

| API | Metal | CUDA | 跨后端通用？ |
|-----|-------|------|-------------|
| `mx.default_device()` | ✅ (返回metal device) | ✅ (返回cuda device) | ✅ |
| `mx.default_stream()` | ✅ | ✅ | ✅ |
| `mx.new_stream()` | ✅ | ✅ | ✅ |
| `mx.set_default_device()` | ✅ | ✅ | ✅ |
| `mx.set_default_stream()` | ✅ | ✅ | ✅ |
| `mx.synchronize()` | ✅ | ✅ | ✅ |
| `mx.device_count()` | ✅ | ✅ | ✅ |
| `mx.device_info()` | ✅ | ✅ | ✅ |

### 3.2 行为差异

**文档说明**：
> "If the stream is unspecified then the operation is run on the default stream of the default device: `mx.default_stream(mx.default_device())`."

这意味着：
- Stream机制在**概念上统一**
- `mx.default_device()` 在Metal上返回metal device，在CUDA上返回cuda device
- 不需要显式指定stream/device，默认行为自动适配当前后端

### 3.3 实际使用建议

```python
# ✅ 跨后端通用，无需分支
default_dev = mx.default_device()
default_stream = mx.default_stream(default_dev)
new_stream = mx.new_stream(default_dev)

# 在操作中指定stream
result = mx.matmul(a, b, stream=new_stream)
```

### 3.4 差异点

**无需 if/else 分支**。Stream API是MLX设计中最统一的部分。

---

## 4. 已知差异或限制

### 4.1 已确认的后端特定问题

| Issue | Metal | CUDA | 影响 |
|-------|-------|------|------|
| QMM (Quantized MatMul) seq_len > 1 | ✅ | ❌ (NYI) | **严重**：量化模型在CUDA上推理失败 (Issue #3122) |
| `mx.fast.metal_kernel` | ✅ | N/A | Metal专用自定义kernel |
| `mx.fast.cuda_kernel` | N/A | ✅ | CUDA专用自定义kernel |
| Metal capture (`start_capture`) | ✅ | N/A | Metal调试专用 |
| Float16 JIT compile on CPU | ✅ | N/A | CPU后端限制 (Issue #3080) |

### 4.2 Quantized Models on CUDA (Issue #3122)

**问题**：CUDA的QMM kernel仅支持 `seq_len=1`

```python
# 在CUDA上会失败
from mlx_vlm import load, generate
model, processor = load("mlx-community/Ministral-3-14B-Instruct-2512-nvfp4")
output = generate(model, processor, prompt="Hi")  # RuntimeError: [QMM] NYI: seq > 1

# 临时解决方案：设置 prefill_step_size=1（仅文本）
output = generate(..., prefill_step_size=1)

# 图像处理仍然失败（multi_modal_projector处理所有patches）
output = generate(..., image="path.jpg")  # 仍失败
```

**影响**：量化模型（nvfp4, mxfp4, mxfp8）在CUDA上的**prefill阶段**和**多模态处理**受限。

### 4.3 `mx.image.resize` 状态

**重要发现**：MLX **没有** `mx.image` 模块在顶层文档中列出，但项目代码实际使用了 `mx.image.resize`：

```python
# backend/engine/runtime/mlx.py
return mx.image.resize(x, (new_H, new_W))
```

这说明：
- `mx.image.resize` **存在且可用**
- 没有搜到CUDA上相关issues，推测跨后端通用
- 但这不是官方文档 prominently 列出的API

### 4.4 `mx.random.categorical` 状态

**搜索结果**：没有找到任何关于 `categorical` 在CUDA上失败的issues。

**结论**：`mx.random.categorical` 在Metal和CUDA上都**正常工作**。

### 4.5 自定义Kernel差异

```python
# Metal专用
mx.fast.metal_kernel(
    name="my_kernel",
    input_names=["x"],
    output_names=["y"],
    source="..."
)

# CUDA专用
mx.fast.cuda_kernel(
    name="my_kernel",
    input_names=["x"],
    output_names=["y"],
    source="..."
)
```

**这是必须分支的API**。

### 4.6 需要 if/else 分支的场景

```python
# 自定义kernel必须分支
if mx.metal.is_available():
    kernel = mx.fast.metal_kernel(...)
elif mx.cuda.is_available():
    kernel = mx.fast.cuda_kernel(...)

# 量化模型推理可能需要分支
if backend == "cuda" and is_quantized_model:
    # 需要特殊处理seq_len=1限制
    prefill_step_size = 1
```

---

## 5. JIT/Compile

### 5.1 `mx.compile` 跨后端支持

| 特性 | Metal | CUDA | 说明 |
|------|-------|------|------|
| `mx.compile()` | ✅ | ✅ | 跨后端通用 |
| 函数缓存 | ✅ | ✅ | 相同shape/dtype不复编译 |
| Shape变化重编译 | ✅ | ✅ | 触发重编译 |
| Dtype变化重编译 | ✅ | ✅ | 触发重编译 |
| `MLX_DISABLE_COMPILE` | ✅ | ✅ | 环境变量通用 |
| `disable_compile()` | ✅ | ✅ | API通用 |

### 5.2 Compile 行为一致性

根据官方文档，`mx.compile`：
1. **跨后端统一API**
2. 第一次调用会trace + optimize + generate + compile（较慢）
3. 后续调用使用缓存（快）
4. 会触发重编译的情况：
   - 输入shape或维度变化
   - 输入类型变化
   - 输入数量变化

### 5.3 已知限制

| 限制 | Metal | CUDA | 说明 |
|------|-------|------|------|
| 不能在compile内print/eval | ✅ | ✅ | 统一限制 |
| 必须是pure function | ✅ | ✅ | 统一限制 |
| Float16 on CPU JIT | ✅ | N/A | CPU后端限制 (Issue #3080) |
| 匿名函数频繁编译 | ✅ | ✅ | 性能陷阱 |

### 5.4 项目中的实际使用

```python
# MLXContext
return mx.compile(fn, *args, **kwargs)

# CudaContext  
return torch.compile(fn, *args, **kwargs)
```

注意：项目中CUDA路径使用 `torch.compile` 而非 `mx.compile`，因为CUDA实现基于PyTorch。

### 5.5 是否需要 if/else 分支？

**使用MLX API时不需要**：`mx.compile` 跨后端通用。

**但项目中需要**：因为CUDA路径使用PyTorch实现，需要 `torch.compile`。

---

## 6. 综合差异汇总

### 6.1 必须 if/else 分支的API

| 场景 | Metal | CUDA | 建议 |
|------|-------|------|------|
| 设备检测 | `mx.metal.is_available()` | `mx.cuda.is_available()` | 分别检测 |
| 内存限制环境变量 | `MLX_METAL_MEMORY_LIMIT` | 无 | 仅Metal设置 |
| 自定义kernel | `mx.fast.metal_kernel` | `mx.fast.cuda_kernel` | 必须分支 |
| 内存统计（项目中） | `mx.get_active_memory()` | `torch.cuda.memory_allocated()` | 使用RuntimeContext封装 |
| 编译（项目中） | `mx.compile` | `torch.compile` | 使用RuntimeContext封装 |

### 6.2 无需分支的API

| API | 说明 |
|-----|------|
| `mx.default_device()` | 自动返回当前后端device |
| `mx.default_stream()` | 自动返回当前后端stream |
| `mx.new_stream()` | 跨后端通用 |
| `mx.set_memory_limit()` | 跨后端通用（但默认行为可能不同） |
| `mx.get_active_memory()` | 跨后端通用 |
| `mx.get_cache_memory()` | 跨后端通用 |
| `mx.clear_cache()` | 跨后端通用 |
| `mx.compile()` | 跨后端通用 |
| `mx.random.categorical` | 跨后端通用 |
| `mx.image.resize` | 跨后端通用（如果可用） |

### 6.3 潜在陷阱

1. **量化模型在CUDA上**：QMM seq_len=1限制可能影响图像生成模型
2. **内存限制默认值**：Metal有默认值（1.5x工作集），CUDA可能没有
3. **`mx.set_memory_limit()` 是软限制**：超出时会使用RAM+swap，不是硬性OOM
4. **环境变量仅Metal**：`MLX_METAL_MEMORY_LIMIT` 在CUDA上完全无效

---

## 7. 对 DanQing Studio 的建议

### 7.1 当前架构的优势

项目中已采用的 **RuntimeContext** 抽象是最佳实践：
- `MLXContext` 处理Metal特定逻辑
- `CudaContext` 处理CUDA/PyTorch特定逻辑
- 上层代码通过 `ctx.backend` 判断后端

### 7.2 需要关注的点

1. **内存限制设置**：当前MLXContext同时设置环境变量和调用API，这是正确的做法
2. **量化模型**：如果未来在CUDA上支持量化模型，需要处理QMM seq_len限制
3. **自定义kernel**：如果模型需要自定义算子，必须提供metal + cuda双版本
4. **图像resize**：`mx.image.resize` 在当前项目中可用，但注意这不是 prominently 文档化的API

### 7.3 代码审查建议

```python
# ✅ 好的做法（已有）
class MLXContext(RuntimeContext):
    def apply_memory_limit_gb(self, gb):
        os.environ["MLX_METAL_MEMORY_LIMIT"] = str(gb)
        mx.set_memory_limit(gb * 1024**3)

# ⚠️ 需要注意
def some_function():
    if mx.metal.is_available():
        # Metal特定逻辑
        pass
    # 缺少 else 分支处理CUDA
```

---

## 8. 参考来源

- MLX官方文档 (v0.31.2): https://ml-explore.github.io/mlx/build/html/index.html
- MLX GitHub Issues: https://github.com/ml-explore/mlx/issues
- 项目代码: `backend/engine/runtime/mlx.py`, `backend/engine/runtime/cuda.py`, `backend/engine/platform.py`
- 关键Issues: #3122 (CUDA QMM), #3080 (CPU JIT Float16), #3350 (Metal cache)
