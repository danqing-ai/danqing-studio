# DanQing-Studio UX-First Optimization Plan
# 用户体验优先：快速出图 + 渐进清晰

> **Version**: v3.0 - UX-First Edition  
> **Date**: 2026-05-23  
> **Core Insight**: Users do not care about "total generation time". They care about "how soon can I see something". 1s blurry sketch >> 5s perfect image.  
> **Design Principle**: Every optimization must reduce "time-to-first-pixel" or improve "clarity-evolution-speed".  
> **Review Target**: Coding Agent  

---

## 1. UX Core Metrics (Not Traditional Performance Metrics)

### 1.1 Traditional vs UX Metrics

| Traditional Metric | UX Metric | Why UX Metric Matters |
|-------------------|-----------|----------------------|
| Total generation time | Time-to-first-pixel (TTFP) | User decides if prompt is correct within 1s |
| Steps per second | Clarity-evolution-speed | User feels "participation" watching image sharpen |
| Throughput | Parameter-response-time | User tweaks 10x more when feedback is instant |
| Memory efficiency | Perceived-latency | Preview stream makes waiting feel shorter |
| Model loading time | Ready-to-create time | User opens app and starts immediately |

### 1.2 Target UX Metrics

| Metric | Current | Target | User Perception |
|--------|---------|--------|-----------------|
| **Time-to-first-pixel (TTFP)** | ~5s | **<=1s** | "I type, I see" |
| **Clarity-evolution-speed** | None (wait for final) | **Preview every 2 steps** | "I watch it come alive" |
| **Parameter-response-time** | ~30s/iteration | **<=2s** | "Like adjusting Photoshop sliders" |
| **Ready-to-create time** | ~10-30s | **<=0s** | "App is always ready" |
| **Video time-to-first-frame** | ~30s | **<=5s** | "Video starts playing immediately" |

---

## 2. Model-Preview Capability Matrix

### 2.1 Preview Types

| Type | Definition | Step Range | User Experience |
|------|-----------|-----------|-----------------|
| **Turbo Preview** | Distilled model, few steps for full sketch | 1-8 NFE | "Type prompt, see complete composition in 1s" |
| **Stream Preview** | Standard model, send intermediate every N steps | 20-50 NFE | "Watch image sharpen step by step" |
| **Frame Preview** | Video model, show frames as generated | Per frame | "Video starts playing while still generating" |
| **No Preview** | Cannot decompose (audio, super-res) | N/A | "Progress bar + play when done" |

### 2.2 Supported Models

| Model Family | Models | Preview Type | Preview Steps | Final Steps | Status |
|-------------|--------|-------------|--------------|-------------|--------|
| **z_image** | z-image-turbo | **Turbo** | **8 NFE** | 28 NFE | Ready |
| **z_image** | z-image | Stream | Every 2 steps | 28-50 NFE | Ready |
| **z_image** | z-image-edit | Stream | Every 2 steps | 50 NFE | Ready |
| **flux** | flux1-schnell | **Turbo** (single-stage) | **4 NFE** | 4 NFE | Ready |
| **flux** | flux2-klein-9b | **Turbo** (single-stage) | **20 NFE** | 20 NFE | Ready |
| **flux** | flux2-klein-4b | **Turbo** (single-stage) | **20 NFE** | 20 NFE | Ready |
| **flux** | flux1-dev | Stream | Every 2 steps | 50 NFE | Ready |
| **ltx** | ltx-video | Frame | Every 2-4 frames | 20-40 NFE | Verify MLX |
| **wan** | wan-video | Frame | Every frame | 50 NFE | Verify MLX |
| **cogvideox** | cogvideox-2b/5b | Frame | Every 2 frames | 50 NFE | Verify MLX |
| **seedvr2** | seedvr2 | **No Preview** | N/A | N/A | Ready |
| **ace_step** | ace-step | **No Preview** | N/A | N/A | Ready |
| **qwen** | qwen3-4b/9b | **No Preview** (encoder) | N/A | N/A | Ready |

### 2.3 Two-Stage vs Single-Stage

**Two-Stage (Preview + Final)**:
- z-image-turbo (8 NFE) -> z-image (28-50 NFE)
- flux1-schnell (4 NFE) -> flux1-dev (50 NFE)

**Single-Stage (Fast enough as final)**:
- flux1-schnell (4 NFE) - quality is acceptable for many use cases
- flux2-klein-9b (20 NFE) - quality near dev, no need for second stage

---

## 3. UX-First Optimization Phases

### Phase 0: The 1-Second Promise (Week 1)

**Goal**: User types prompt, sees first pixel in <=1 second.

#### 3.0.1 Turbo Model Always-Resident

```python
# backend/engine/runtime/ux_first_resident.py
class UXFirstResidentManager:
    """
    UX-First: Turbo models are ALWAYS resident.
    Non-Turbo models load on-demand.
    """

    TURBO_MODELS = ["z-image-turbo", "flux1-schnell", "flux2-klein-9b"]

    def __init__(self):
        self._turbo_resident = {}  # Always loaded
        self._on_demand = {}       # Load when needed

    def startup(self):
        """Called once at app launch."""
        for model_id in self.TURBO_MODELS:
            if self._is_installed(model_id):
                self._turbo_resident[model_id] = self._load(model_id)

        logger.info(f"Turbo models resident: {list(self._turbo_resident.keys())}")

    def get_turbo(self, family: str) -> dict:
        """Instant return, guaranteed resident."""
        # Map family to turbo model
        turbo_map = {
            "z_image": "z-image-turbo",
            "flux": "flux1-schnell",
        }
        model_id = turbo_map.get(family)
        if model_id and model_id in self._turbo_resident:
            return self._turbo_resident[model_id]
        raise RuntimeError(f"No turbo model available for {family}")
```

**Why this matters**: If turbo model is not resident, TTFP is 10-30s (model loading). With resident, TTFP is ~1s.

---

#### 3.0.2 Fast-First VAE Decode

```python
# backend/engine/pipelines/fast_first_vae.py
class FastFirstVAEDecoder:
    """
    UX-First: First preview at 256px, then upscale.
    User sees SOMETHING immediately, quality improves gradually.
    """

    def decode_progressive(self, latents: mx.array, step: int, total_steps: int) -> bytes:
        """
        Step 1-2: 64px thumbnail (fastest)
        Step 3-4: 256px preview (good enough to see composition)
        Step 5+: 512px preview (details emerging)
        Final: Full resolution
        """
        if step <= 2:
            size = 64
        elif step <= 4:
            size = 256
        elif step <= total_steps * 0.8:
            size = 512
        else:
            size = None  # Full resolution

        return self._decode_at_size(latents, size)

    def _decode_at_size(self, latents: mx.array, size: int | None) -> bytes:
        if size is None:
            pixels = self.vae.decode(latents)
        else:
            # Fast path: decode at smaller size
            down_latents = mx.interpolate(latents, scale_factor=0.5, mode="nearest")
            pixels = self.vae.decode(down_latents)
            pixels = mx.interpolate(pixels, size=(size, size), mode="bilinear")

        return self._to_png(pixels)
```

**Why this matters**: VAE decode is expensive. 64px decode is ~10x faster than 1024px. User sees thumbnail in <100ms.

---

#### 3.0.3 Precompute Everything

```python
# backend/engine/families/z_image/transformer.py
class UXFirstTransformer(TransformerBase):
    """
    UX-First: Precompute ALL expensive operations at model load time.
    Runtime path must be: load weights -> matrix multiply -> done.
    """

    def __init__(self, config, ctx: RuntimeContext):
        super().__init__(config, ctx)
        self._precompute_rope()
        self._precompute_position_embeddings()
        self._jit_compile_forward()

    def _precompute_rope(self):
        max_seq_len = self.config.max_position_embeddings
        dim = self.config.head_dim
        inv_freq = 1.0 / (self.config.rope_theta ** (mx.arange(0, dim, 2) / dim))
        t = mx.arange(max_seq_len)
        freqs = mx.outer(t, inv_freq)
        self.rope_cos = mx.cos(freqs)
        self.rope_sin = mx.sin(freqs)

    def _jit_compile_forward(self):
        """Compile forward pass to reduce dispatch overhead."""
        self._compiled_forward = mx.compile(self._forward_impl)
```

**Why this matters**: Every ms saved in per-step overhead directly improves TTFP.

---

### Phase 1: The Living Canvas (Week 2-3)

**Goal**: Canvas updates like a living thing - every 2 steps, image gets sharper.

#### 3.1.1 Aggressive Preview Stream

```python
# backend/engine/pipelines/living_canvas.py
class LivingCanvasPipeline(ImagePipeline):
    """
    UX-First: Stream preview EVERY 2 STEPS.
    Personal tool has no server cost - be aggressive.
    """

    def run(self, request: ImageRequest):
        latents = self._init_noise(request.seed)

        for i, t in enumerate(request.scheduler.timesteps):
            # Denoise step
            noise_pred = self.model(latents, t)
            latents = request.scheduler.step(latents, noise_pred, t)

            # Preview EVERY 2 steps (not every 5)
            if i % 2 == 0 or i == len(request.scheduler.timesteps) - 1:
                preview = self._fast_decode(latents, step=i, total=len(request.scheduler.timesteps))
                yield PreviewEvent(image=preview, step=i, total=len(request.scheduler.timesteps))

        # Final decode at full resolution
        final = self.vae.decode(latents)
        yield FinalEvent(image=final)
```

**Why this matters**: Every 2 steps = 4x more preview frames than every 8 steps. User feels "participation".

---

#### 3.1.2 Resolution Escalation

```python
# frontend/src/components/LivingCanvas.vue
<template>
  <div class="canvas-container">
    <!-- Layer 1: 64px thumbnail (appears at step 1) -->
    <img v-if="preview64" :src="preview64" class="layer thumbnail" />

    <!-- Layer 2: 256px preview (appears at step 3) -->
    <img v-if="preview256" :src="preview256" class="layer preview" 
         :class="{active: preview256 && !preview512}" />

    <!-- Layer 3: 512px preview (appears at step 6) -->
    <img v-if="preview512" :src="preview512" class="layer preview" 
         :class="{active: preview512 && !final}" />

    <!-- Layer 4: Final image (appears at end) -->
    <img v-if="final" :src="final" class="layer final" />
  </div>
</template>
```

**Why this matters**: User sees image "grow" from thumbnail to masterpiece. Creates emotional connection.

---

### Phase 2: The Instant Feedback Loop (Week 4-6)

**Goal**: Adjust parameter, see change in <=2 seconds.

#### 3.2.1 Latent Snapshot System

```python
# backend/engine/pipelines/snapshot_pipeline.py
class SnapshotPipeline(ImagePipeline):
    """
    UX-First: Save latent snapshots at key steps.
    Parameter change = resume from nearest snapshot, not restart.
    """

    def __init__(self):
        self._snapshots = {}  # request_id -> {step: latent}
        self._snapshot_interval = 5  # Save every 5 steps

    def run(self, request: ImageRequest):
        request_id = request.id

        # Check if we have snapshots for similar request
        base_request = request.without_parameters(["cfg", "steps"])
        snapshots = self._snapshots.get(base_request.fingerprint)

        if snapshots and request.step > 0:
            # Resume from nearest snapshot
            nearest_step = max(s for s in snapshots.keys() if s <= request.step)
            latents = snapshots[nearest_step]
            start_idx = list(request.scheduler.timesteps).index(nearest_step)
        else:
            # Fresh start
            latents = self._init_noise(request.seed)
            start_idx = 0

        for i in range(start_idx, len(request.scheduler.timesteps)):
            t = request.scheduler.timesteps[i]
            noise_pred = self.model(latents, t, cfg=request.cfg)
            latents = request.scheduler.step(latents, noise_pred, t)

            # Save snapshot
            if i % self._snapshot_interval == 0:
                if request_id not in self._snapshots:
                    self._snapshots[request_id] = {}
                self._snapshots[request_id][i] = latents.copy()

            # Preview
            if i % 2 == 0:
                yield PreviewEvent(image=self._fast_decode(latents), step=i)

        yield FinalEvent(image=self.vae.decode(latents))
```

**Why this matters**: User drags CFG slider from 5 to 7. Instead of 30s restart, system resumes from step 20 snapshot and only runs steps 21-50. Response time: ~2s.

---

#### 3.2.2 Parameter Diff Tracker

```python
# backend/engine/pipelines/diff_tracker.py
class ParameterDiffTracker:
    """
    UX-First: Track which parameters changed.
    Only re-run affected computation.
    """

    CHANGE_IMPACT = {
        "prompt": "full",        # Must re-encode text
        "seed": "full",          # Must restart
        "cfg": "partial",        # Can resume from snapshot
        "steps": "partial",      # Can resume if increasing
        "width": "full",         # Must restart
        "height": "full",        # Must restart
        "model": "full",         # Must restart
    }

    def analyze_change(self, old_request: ImageRequest, new_request: ImageRequest) -> dict:
        changes = {}
        for param in ["prompt", "seed", "cfg", "steps", "width", "height", "model"]:
            if getattr(old_request, param) != getattr(new_request, param):
                changes[param] = self.CHANGE_IMPACT[param]

        if all(impact == "partial" for impact in changes.values()):
            return {"strategy": "resume", "from_step": self._find_resume_point(old_request)}
        else:
            return {"strategy": "restart"}
```

**Why this matters**: User tweaks CFG 10 times. 9 times = partial change = 2s response. 1 time = changes prompt = 30s restart. Average: ~5s instead of 30s.

---

### Phase 3: Video Becomes Alive (Week 6-8)

**Goal**: Video generation feels like "film developing" - frames appear as they generate.

#### 3.3.1 Frame-By-Frame Preview

```python
# backend/engine/pipelines/video_living_canvas.py
class VideoLivingCanvasPipeline(VideoPipeline):
    """
    UX-First: Show frames as they generate.
    Like watching film develop in darkroom.
    """

    def run(self, request: VideoRequest):
        # Generate latent frames
        frame_latents = self._generate_frame_latents(request)

        # Decode and show frames progressively
        for i in range(0, len(frame_latents), self.CHUNK_SIZE):
            chunk = frame_latents[i:i+self.CHUNK_SIZE]
            frames = self.vae.decode(chunk)

            for frame in frames:
                yield FrameEvent(frame=frame, frame_number=i, total=len(frame_latents))

            # Clean up to prevent memory bloat
            del chunk, frames
            mx.eval()
```

**Why this matters**: 10s video at 30fps = 300 frames. User sees frame 1 at 5s instead of waiting 300s for all frames.

---

## 4. UI/UX Integration

### 4.1 The Living Canvas Component

```
┌─────────────────────────────────────────────────────────────┐
│  Prompt: [A cat in space helmet...                    ]     │
│                                                              │
│  ┌─────────────────────────────┐  ┌──────────────────────┐  │
│  │                             │  │ Quality: [Fast ▼]    │  │
│  │    🖼️ Living Canvas         │  │ Steps:   [8]         │  │
│  │                             │  │ CFG:     [5.0]       │  │
│  │    [Thumbnail -> 256px      │  │ Seed:    [Random ▼]  │  │
│  │     -> 512px -> Final]      │  │                      │  │
│  │                             │  │ [Generate] [Enhance] │  │
│  │    Step 12/50...            │  │                      │  │
│  │                             │  │                      │  │
│  └─────────────────────────────┘  └──────────────────────┘  │
│                                                              │
│  Memory: [████████░░░░░░░░] 12GB / 32GB (38%)               │
│  Status: 🟢 Turbo resident | 🟡 Dev on-demand               │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Parameter Slider Interaction

```
CFG Slider: [1----|----5----|----20]
            ↑     ↑         ↑
            1s    2s        2s

User drags slider -> releases -> sees preview update in 2s
(Not: drags -> releases -> waits 30s -> sees result)
```

---

## 5. Implementation Checklist (UX-First)

| Optimization | UX Metric | Test Method | Acceptance |
|--------------|-----------|-------------|------------|
| Turbo resident | TTFP <=1s | Stopwatch from Enter key to first pixel | Must feel instant |
| Fast-first VAE | TTFP <=1s | Profile 64px decode time | <100ms |
| Living canvas | Clarity-evolution | User survey: "Did you feel engaged?" | >80% yes |
| Snapshot system | Parameter-response | Drag slider 10 times, measure average | <5s average |
| Video frame preview | Video-TTFP | Time to first frame | <5s |

---

## 6. What We Sacrifice for UX

| Traditional Optimization | UX-First Choice | Why |
|---------------------------|-----------------|-----|
| Memory efficiency (unload models) | Keep all models resident | TTFP > memory |
| Batch processing (merge requests) | Never batch, always instant | Response > throughput |
| Full-resolution preview | 64px -> 256px -> 512px -> final | Speed > initial quality |
| Disk space (cache cleanup) | Keep all history forever | User wants to compare versions |
| CPU efficiency (preview decode on GPU) | Decode on CPU to free GPU | GPU is for generation, CPU for preview |

---

> **Coding Agent Review Checklist (UX-First)**:
> 1. Does every optimization reduce "time-to-first-pixel" or improve "clarity-evolution-speed"?
> 2. Are Turbo models ALWAYS resident, never unloaded?
> 3. Does preview stream EVERY 2 steps, not every 5 or 8?
> 4. Can user adjust parameter and see change in <5s average?
> 5. Does video show frames as they generate, not wait for completion?
> 6. Is there a 64px thumbnail path that decodes in <100ms?
