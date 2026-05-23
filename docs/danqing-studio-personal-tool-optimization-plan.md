# DanQing-Studio Performance Optimization Plan (Personal Creative Tool Edition)

> **Version**: v2.0 - Refreshed for Personal Creative Tool  
> **Date**: 2026-05-23  
> **Goal**: Optimize for single-user local creative workflow: instant response, real-time preview, interactive parameter tuning. No multi-user concurrency concerns.  
> **Review Target**: Coding Agent  

---

## 1. Product Positioning Refresh

### 1.1 From "Service" to "Personal Tool"

| Dimension | Multi-User Service (Previous Plan) | Personal Creative Tool (This Plan) |
|-----------|--------------------------------------|-----------------------------------|
| **Users** | Multiple users, remote access | Single user, local machine |
| **Concurrency** | High concurrency, needs scheduling | No concurrency, serial execution |
| **Queue** | Global queue with priority | No queue, direct execution |
| **Memory** | Strict isolation, multi-tenant | Full machine RAM available |
| **Network** | SSE latency sensitive | Local API, negligible latency |
| **Cache** | Limited, shared across users | Aggressive caching, abundant disk |
| **Model Loading** | On-demand, LRU eviction | Resident in memory, never unload |
| **Preview** | Optional, server-side cost | Mandatory, real-time creative feedback |
| **Quality Strategy** | Uniform quality | Turbo preview + Standard final |
| **Optimization Goal** | Throughput, fairness, utilization | Response speed, interaction fluidity, creative experience |

### 1.2 Removed Optimizations (Service-Only)

| Removed Item | Reason |
|--------------|--------|
| Multi-Queue Scheduler | Single user has no concurrent tasks |
| Dynamic Batch Merge | No batch requests from single user |
| Result Fingerprint Cache | Personal tool keeps history directly, no need for fingerprint dedup |
| Async Model Warmup API | Models are resident, no warmup needed |
| Cross-Family Component Sharing | Resident models already solve this |
| Memory-Mapped Weight Loading | One-time load at startup is sufficient |

### 1.3 New Optimizations (Personal-Tool-Specific)

| New Item | Why Personal Tool Needs It |
|----------|---------------------------|
| Turbo Real-Time Preview | Creative workflow requires instant visual feedback |
| Parameter Real-Time Response | Adjust cfg/steps/seed and see effect immediately |
| History Version Comparison | Side-by-side comparison of different parameter sets |
| Memory Pressure UI Indicator | User needs to know GPU memory status to avoid OOM |
| Model On-Demand Resident | Only installed models load at startup, not all |

---

## 2. Optimization Targets (Personal Tool)

| Metric | Current | Target | User Perception |
|--------|---------|--------|-----------------|
| **Preview latency** | ~5s (full generation) | **<=1s** (Turbo preview) | "Type prompt, see result instantly" |
| **Final image latency** | ~5s | **<=3s** | "Click generate, get final image quickly" |
| **First interaction** | ~10-30s (model loading) | **<=0s** (resident) | "Open app, ready to create immediately" |
| **Edit/retouch latency** | ~5-6s | **<=3s** | "Adjust reference image, see result fast" |
| **Parameter tuning cycle** | ~30s per iteration | **<=5s** | "Tweak parameters, see effect in real-time" |
| **Video memory safety** | May OOM on long video | **Never OOM** | "Generate long video without crash" |

---

## 3. Optimization Details

### Phase 1: Instant Creative Experience (Week 1)

#### 3.1.1 Turbo Real-Time Preview (P0)

**Problem**: User types prompt, waits ~5s for full generation before seeing anything.  
**Solution**: Two-stage pipeline: Stage 1 = Turbo (8 steps) instant preview, Stage 2 = Standard (50 steps) final output on user confirmation.

```python
# backend/engine/pipelines/image_pipeline.py
class TwoStageImagePipeline(ImagePipeline):
    def run(self, request: ImageRequest) -> Generator[PipelineEvent, None, None]:
        # Stage 1: Turbo preview (always run)
        turbo_request = request.with_model("z-image-turbo").with_steps(8)
        preview_latents = self._denoise(turbo_request)
        preview_image = self._vae_decode(preview_latents, target_size=512)
        yield PipelineEvent(type="preview", image=preview_image, stage=1)

        # Stage 2: Standard final (on user confirmation or auto-promote)
        if request.quality_mode == "final":
            standard_request = request.with_model("z-image").with_steps(50)
            final_latents = self._denoise(standard_request)
            final_image = self._vae_decode(final_latents, target_size=request.resolution)
            yield PipelineEvent(type="final", image=final_image, stage=2)
```

**Frontend Integration**:
- Create tab: Show Turbo preview within 1s of typing
- "Enhance" button: Promote preview to final quality
- Auto-enhance: After 3s idle, auto-run final quality

**Gain**: **Preview latency ~1s** (from ~5s)  
**Difficulty**: Low - Registry + Pipeline modification  
**Quality Tradeoff**: Turbo PSNR 16.7 dB is acceptable for preview, not for final. UI clearly labels "Preview (fast)" vs "Final (high quality)".

---

#### 3.1.2 Model Resident Memory (P0)

**Problem**: "First run is slow - model load into GPU memory" (AGENTS.md).  
**Solution**: At app startup, load ALL installed models into GPU memory. Never unload.

```python
# backend/engine/runtime/resident_manager.py
class ModelResidentManager:
    """Personal tool: all installed models resident in GPU, never unload."""

    def __init__(self, models_dir: Path = Path("./models")):
        self.models_dir = models_dir
        self._resident = {}  # model_id -> weights
        self._memory_gb = 0.0

    def startup_load_all(self):
        """Called once at app startup."""
        installed = self._scan_installed_models()
        for model_id in installed:
            try:
                weights = self._load_model(model_id)
                self._resident[model_id] = weights
                self._memory_gb += self._measure_memory(weights)
            except Exception as e:
                # Log but don't block startup
                logger.warning(f"Failed to resident {model_id}: {e}")

        logger.info(f"Resident models: {len(self._resident)}, Memory: {self._memory_gb:.1f}GB")

    def get(self, model_id: str) -> dict:
        """Instant return, no loading."""
        if model_id not in self._resident:
            raise RuntimeError(f"Model {model_id} not resident. Install first.")
        return self._resident[model_id]

    def _scan_installed_models(self) -> list[str]:
        """Scan models/ directory for installed model families."""
        # Check model_index.json or safetensors files
        return [...]
```

**Startup Sequence**:
1. Tauri desktop app launches
2. Sidecar (Python backend) starts
3. `ModelResidentManager.startup_load_all()` runs in background thread
4. UI shows "Loading models..." progress
5. Once loaded, UI ready. All subsequent generations are instant.

**Gain**: **First interaction ~0s** (from ~10-30s)  
**Difficulty**: Low - Startup sequence modification  
**Memory Constraint**: If total model size > `MLX_METAL_MEMORY_LIMIT`, show warning in UI: "Models exceed GPU memory. Please uninstall unused models in Settings."

---

### Phase 2: Interactive Tuning (Week 2-3)

#### 3.2.1 Real-Time Preview Stream (P0)

**Problem**: User waits for full generation before seeing any intermediate result.  
**Solution**: Send low-resolution preview every 2-3 steps via SSE. Personal tool has no server pressure, can stream aggressively.

```python
# backend/engine/pipelines/preview_stream.py
class PreviewStreamDecoder:
    """Aggressive preview for personal tool: every 2 steps."""

    def __init__(self, vae, preview_size: int = 256):
        self.vae = vae
        self.preview_size = preview_size
        self._step_counter = 0

    def maybe_decode(self, latents: mx.array) -> bytes | None:
        self._step_counter += 1
        if self._step_counter % 2 != 0:  # Every 2 steps
            return None

        # Fast decode on CPU to avoid blocking GPU
        with mx.stream(mx.cpu):
            preview = self._fast_vae_decode(latents, target_size=self.preview_size)
            return self._to_png_bytes(preview)

    def _fast_vae_decode(self, latents, target_size: int):
        """Use VAE at 1/4 resolution for speed."""
        # Downsample latents before decode
        down_latents = mx.interpolate(latents, scale_factor=0.5, mode="nearest")
        pixels = self.vae.decode(down_latents)
        return mx.interpolate(pixels, size=(target_size, target_size), mode="bilinear")
```

**Frontend**: Canvas shows preview updating in real-time like a progress bar with actual image content.  
**Gain**: **Creative experience质变** - user can abort early if direction is wrong  
**Difficulty**: Medium - Fast VAE decode path  
**Personal Tool Advantage**: No server cost concern, can decode every step if GPU allows.

---

#### 3.2.2 Reference Image VAE Cache (P1)

**Problem**: Edit/retouch/extend re-encodes reference image every time.  
**Solution**: File hash -> latent permanent cache. Personal tool has abundant disk.

```python
# backend/engine/pipelines/reference_cache.py
class ReferenceImageCache:
    """Personal tool: permanent cache, no eviction."""

    CACHE_DIR = Path("~/Library/Caches/DanQingStudio/vae_latents")  # macOS

    def get(self, image_path: Path, vae_config: dict) -> mx.array | None:
        h = self._file_hash(image_path)
        cache_file = self.CACHE_DIR / f"{h}_{vae_config.get('scaling_factor',1.0)}.npz"
        if cache_file.exists():
            return mx.load(str(cache_file))
        return None

    def put(self, image_path: Path, vae_config: dict, latent: mx.array):
        h = self._file_hash(image_path)
        cache_file = self.CACHE_DIR / f"{h}_{vae_config.get('scaling_factor',1.0)}.npz"
        mx.save(str(cache_file), latent)

    def _file_hash(self, path: Path) -> str:
        import hashlib
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
```

**Gain**: **Edit latency ~3s** (from ~5-6s)  
**Difficulty**: Low - Pipeline layer addition  
**Cleanup**: No cleanup needed for personal tool. Cache grows with usage, user can clear in Settings.

---

#### 3.2.3 Conditioning Cache (P1)

**Problem**: Same prompt re-encodes text every time.  
**Solution**: Unlimited in-memory cache (single user, no memory pressure from other users).

```python
# backend/engine/common/conditioning_cache.py
class UnlimitedConditioningCache:
    """Personal tool: no size limit, keep all embeddings in memory."""

    def __init__(self):
        self._cache = {}  # key -> mx.array

    def get(self, prompt: str, model_family: str) -> mx.array | None:
        key = f"{model_family}:{prompt}"
        return self._cache.get(key)

    def put(self, prompt: str, model_family: str, embeds: mx.array):
        key = f"{model_family}:{prompt}"
        self._cache[key] = embeds
```

**Gain**: **~10-15%** when tweaking prompt (e.g., adding "8k, highly detailed" to existing prompt)  
**Difficulty**: Low  
**Scope**: All model families.

---

#### 3.2.4 Precompute RoPE (P1)

**Problem**: Z-Image/Qwen 3D RoPE recalculates every step.  
**Solution**: Precompute at model load time (resident models only compute once).

```python
# backend/engine/families/z_image/transformer.py
class ZImageTransformer(TransformerBase):
    def __init__(self, config, ctx: RuntimeContext):
        super().__init__(config, ctx)
        self._precompute_rope()

    def _precompute_rope(self):
        max_seq_len = self.config.max_position_embeddings
        dim = self.config.head_dim
        inv_freq = 1.0 / (self.config.rope_theta ** (mx.arange(0, dim, 2) / dim))
        t = mx.arange(max_seq_len)
        freqs = mx.outer(t, inv_freq)
        self.rope_cos = mx.cos(freqs)
        self.rope_sin = mx.sin(freqs)

    def _apply_rope(self, x: mx.array, positions: mx.array) -> mx.array:
        cos = self.rope_cos[positions]
        sin = self.rope_sin[positions]
        return x * cos + self._rotate_half(x) * sin
```

**Gain**: **~15-20%** for Z-Image/Qwen/fibo  
**Difficulty**: Medium  
**OminiX-MLX Reference**: `zimage-mlx` `compute_rope()` pattern.

---

#### 3.2.5 Attention Path Verification (P1)

**Problem**: May not use optimal SDPA.  
**Solution**: Verify `mx.fast.scaled_dot_product_attention` is called.

```python
# backend/engine/runtime/mlx.py
class MLXContext(RuntimeContext):
    def attention(self, query, key, value, mask=None):
        if hasattr(mx.fast, 'scaled_dot_product_attention'):
            return mx.fast.scaled_dot_product_attention(
                query, key, value, 
                scale=self._compute_scale(query),
                mask=mask
            )
        return self._manual_attention(query, key, value, mask)
```

**Verification**: Add profile logging in dev mode.  
**Gain**: **~20-40%** for long sequences  
**Difficulty**: Low.

---

### Phase 3: Advanced Interaction (Week 4-6)

#### 3.3.1 Parameter Real-Time Response (P2)

**Problem**: Adjusting cfg/steps/seed requires full regeneration (~30s per iteration).  
**Solution**: Cache intermediate latents, only re-run scheduler/noise part.

```python
# backend/engine/pipelines/interactive_pipeline.py
class InteractivePipeline(ImagePipeline):
    def __init__(self):
        self._latent_cache = {}  # request_hash -> intermediate latents
        self._noise_cache = {}   # seed -> initial noise

    def generate_with_param_change(
        self, 
        base_request: ImageRequest,
        param_changes: dict,  # e.g., {"cfg": 7.0, "steps": 30}
        cached_latents: mx.array
    ) -> mx.array:
        """
        Reuse cached latents from step N/2,
        only re-run denoise from there with new params.
        """
        # Find restart point (e.g., step 25 of 50)
        restart_step = len(cached_latents) // 2

        # Re-run from restart_step with new cfg/steps
        new_request = base_request.with_overrides(param_changes)
        latents = cached_latents[restart_step]

        for t in new_request.scheduler.timesteps[restart_step:]:
            noise_pred = self.model(latents, t, cfg=new_request.cfg)
            latents = new_request.scheduler.step(latents, noise_pred, t)

        return latents
```

**Frontend**: Slider for cfg (1-20), steps (1-50). Drag slider, see preview update in ~2s.  
**Gain**: **Parameter tuning cycle ~5s** (from ~30s)  
**Difficulty**: High - Need to cache and resume from intermediate state  
**Constraint**: Only works for increasing steps or cfg changes, not seed changes (seed affects initial noise).

---

#### 3.3.2 History Version Comparison (P2)

**Problem**: User generates multiple variants, hard to compare.  
**Solution**: Side-by-side comparison of all versions of same prompt.

```python
# backend/persistence/version_store.py
class VersionStore:
    """Store all generations of same prompt for comparison."""

    def save_version(self, prompt: str, params: dict, asset_id: str):
        version_group = self._get_version_group(prompt)
        version_group.versions.append({
            "params": params,
            "asset_id": asset_id,
            "timestamp": time.time()
        })
        self._save(version_group)

    def get_comparison_view(self, prompt: str) -> list[dict]:
        """Return all versions for side-by-side display."""
        group = self._get_version_group(prompt)
        return sorted(group.versions, key=lambda v: v["timestamp"])
```

**Frontend**: Gallery tab shows "Versions" button. Click opens split-screen view with all variants.  
**Gain**: **Creative decision efficiency** - no more manual file comparison  
**Difficulty**: Low - SQLite schema + frontend component.

---

#### 3.3.3 Memory Pressure UI Indicator (P1)

**Problem**: User doesn't know GPU memory status, may OOM unexpectedly.  
**Solution**: Real-time memory indicator in UI top bar.

```python
# backend/api/routes/system.py
@router.get("/api/system/memory")
def get_memory_status():
    active_gb = mx.metal.get_active_memory() / 1e9
    limit_gb = float(os.environ.get("MLX_METAL_MEMORY_LIMIT", 120))
    return {
        "active_gb": active_gb,
        "limit_gb": limit_gb,
        "usage_percent": active_gb / limit_gb * 100,
        "resident_models": list(resident_manager._resident.keys())
    }
```

**Frontend**: Top bar shows memory bar (green < 50%, yellow 50-80%, red > 80%). Hover shows model list.  
**Gain**: **User awareness, prevents OOM**  
**Difficulty**: Low.

---

### Phase 4: Video & Stability (Week 6-8)

#### 3.4.1 Progressive VAE Decode (P2)

**Problem**: Long video one-shot decode causes OOM on personal machine.  
**Solution**: Chunk-based decode with explicit cleanup.

```python
# backend/engine/common/vae/progressive_decoder.py
class ProgressiveVAEDecoder:
    def __init__(self, vae, chunk_frames: int = 8):
        self.vae = vae
        self.chunk_frames = chunk_frames

    def decode(self, latents: mx.array) -> list:
        B, C, T, H, W = latents.shape
        frames = []

        for i in range(0, T, self.chunk_frames):
            chunk = latents[:, :, i:i+self.chunk_frames, :, :]
            chunk_pixels = self.vae.decode(chunk)
            for b in range(B):
                for t in range(chunk_pixels.shape[2]):
                    frame = chunk_pixels[b, :, t, :, :]
                    frames.append(self._to_pil(frame))

            del chunk, chunk_pixels
            mx.eval()  # Force MLX GC

        return frames
```

**Gain**: **Never OOM on long video**  
**Difficulty**: Medium  
**Applicable**: `ltx`, `wan`, `cogvideox`.

---

#### 3.4.2 Model On-Demand Resident (P1)

**Problem**: Startup loads ALL models, may exceed memory if user has many installed.  
**Solution**: Only load models that are actually installed, lazy load on first use if skipped at startup.

```python
# backend/engine/runtime/resident_manager.py
class SmartResidentManager:
    def startup_load_priority(self):
        """Load priority models first, others on-demand."""
        priority = ["z-image-turbo", "flux2-klein-9b"]  # Default + most used

        for model_id in priority:
            if self._is_installed(model_id):
                self._resident(model_id)

        # Others loaded on first use
        self._on_demand_models = self._get_installed_except(priority)

    def get(self, model_id: str) -> dict:
        if model_id in self._resident:
            return self._resident[model_id]

        # Lazy load
        if model_id in self._on_demand_models:
            self._resident(model_id)
            return self._resident[model_id]

        raise RuntimeError(f"Model {model_id} not installed")
```

**Gain**: **Startup memory controlled** - only hot models pre-loaded  
**Difficulty**: Low.

---

## 4. OminiX-MLX Architecture Borrowing (Personal Tool Context)

### 4.1 What We Borrow

| OminiX-MLX Pattern | DanQing Personal Tool Adaptation | Why Different |
|---------------------|----------------------------------|---------------|
| `compute_rope()` | Precompute once at model load | No need to optimize for multi-request sharing |
| Model LRU cache | Permanent resident, no eviction | Single user, memory is dedicated |
| Batch inference | Not applicable | No concurrent requests |
| Stream SSE | Aggressive preview every 2 steps | No server cost, maximize user experience |
| `ominix-api` multi-thread | Not applicable | Single user, single request at a time |

### 4.2 What We Don't Need

| OminiX-MLX Feature | Why Not Needed in Personal Tool |
|-------------------|--------------------------------|
| Multi-thread request handling | Single user, serial workflow |
| Request queue | Direct execution, no queuing |
| Batch aggregation | No batch requests |
| Model eviction/LRU | Resident forever |
| Cross-request deduplication | Single user, no duplicate requests |

---

## 5. Implementation Checklist

### 5.1 Code Standards (AGENTS.md Constraints)

| Check Item | Requirement | CI Command |
|------------|-------------|------------|
| Zero leak | `import mlx` / `import torch` only in `runtime/` and `*_mlx.py` / `*_cuda.py` | `make check-engine-imports` |
| Plugin | New optimizations do not introduce Pipeline `family` branches | `make check-consistency` |
| Contract | All API changes through contracts | Manual check |

### 5.2 Testing (Personal Tool Focus)

| Optimization | Test Method | Acceptance Criteria |
|--------------|-------------|---------------------|
| Turbo preview | Manual UI test | Preview appears <=1s after prompt input |
| Model resident | Startup timing | App ready <=5s after launch (models loading in background) |
| Real-time preview stream | Visual test | Preview updates every 2 steps, no stutter |
| Parameter real-time response | Slider drag test | Preview updates <=2s after slider release |
| Reference image cache | Edit repeat test | Second edit <=3s, first edit <=5s |
| Memory indicator | Memory pressure test | Indicator accurate within 10% |
| Progressive VAE | Long video test | 10s video generates without OOM on 32GB Mac |

### 5.3 Benchmark

```bash
# Personal tool benchmark: focus on interactive latency
make bench-setup
make bench-first-interaction    # Time from app launch to first generation ready
make bench-preview-latency      # Time from prompt to preview
make bench-final-latency        # Time from confirm to final image
make bench-edit-latency         # Time from reference image to edit result
make bench-param-response       # Time from slider change to preview update
make bench-video-memory         # Peak memory during 10s video generation
```

---

## 6. UI/UX Integration

### 6.1 Create Tab Redesign

```
┌─────────────────────────────────────────────────────────┐
│  Prompt Input                                            │
│  [A cat sitting on a windowsill...        ] [Generate]  │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Preview (Turbo) │  │ Parameters                 │  │
│  │                 │  │  Quality: [Fast ▼]         │  │
│  │  [Image]        │  │  Steps:   [8----|----50]   │  │
│  │  0.8s           │  │  CFG:     [1----|----20]    │  │
│  │                 │  │  Seed:    [Random ▼]        │  │
│  │  [Enhance]      │  │                            │  │
│  │                 │  │  [Generate Final]          │  │
│  └─────────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│  Memory: [████████░░░░░░░░] 12GB / 32GB (38%)          │
└─────────────────────────────────────────────────────────┘
```

### 6.2 Memory Indicator

- **Green** (< 50%): All models resident, full performance
- **Yellow** (50-80%): Some models may need lazy load
- **Red** (> 80%): Warning - uninstall unused models or reduce quality

---

## 7. Milestones (Personal Tool)

| Phase | Time | Deliverables | User Experience |
|-------|------|--------------|-----------------|
| **Phase 1** | Week 1 | Turbo preview + Model resident | "Type prompt, see preview in 1 second" |
| **Phase 2** | Week 2-3 | Real-time stream + Reference cache + Conditioning cache + RoPE precompute | "Edit feels instant, preview streams smoothly" |
| **Phase 3** | Week 4-6 | Parameter real-time response + History comparison + Memory indicator | "Tweak parameters like adjusting Photoshop sliders" |
| **Phase 4** | Week 6-8 | Progressive VAE + Smart resident | "Generate 10s video without worrying about crash" |

---

## 8. Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Turbo preview quality too low | UI clearly labels "Preview (draft quality)", user expects rough result |
| Model resident exceeds memory | Smart resident loads priority models only, others lazy load; UI warns |
| Real-time preview too CPU-heavy | Preview every 2-3 steps, not every step; downscale to 256px |
| Parameter response breaks determinism | Document that real-time response is approximate, final generation is exact |
| Cache disk grows unbounded | Settings page shows cache size, one-click clear |

---

## 9. Appendix

### 9.1 Reference Projects

| Project | Link | Borrowing Point |
|---------|------|-----------------|
| OminiX-MLX | github.com/OminiX-ai/OminiX-MLX | Precompute RoPE, model resident pattern |
| ComfyUI | github.com/comfyanonymous/ComfyUI | Real-time preview mechanism |
| Stable Diffusion WebUI | github.com/AUTOMATIC1111/stable-diffusion-webui | Parameter slider interaction |

### 9.2 Environment Variables

| Variable | Description | Personal Tool Usage |
|----------|-------------|---------------------|
| `MLX_METAL_MEMORY_LIMIT` | GPU memory limit | Set to 80% of system RAM for safety |
| `DANQING_RESIDENT_MODELS` | Comma-separated priority models | `z-image-turbo,flux2-klein-9b` |
| `DANQING_PREVIEW_INTERVAL` | Steps between previews | `2` for aggressive, `5` for conservative |
| `DANQING_CACHE_DIR` | Cache directory | `~/Library/Caches/DanQingStudio` |

---

> **Coding Agent Review Checklist (Personal Tool)**:
> 1. Does every optimization prioritize single-user interactive experience over throughput?
> 2. Are models resident and never unloaded (no LRU eviction)?
> 3. Is preview streaming aggressive (every 2-3 steps) without server-cost concern?
> 4. Does UI provide real-time feedback (memory, progress, preview)?
> 5. Are all caches permanent (no TTL) with user-controlled cleanup?
