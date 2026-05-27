"""HeartMuLa text-to-music — MLX inference (loads MLX weights from download-time conversion)."""
from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, List, Optional

import mlx.core as mx
import numpy as np

from backend.engine.common.mlx_runtime_fallback import set_random_seed
from backend.engine.common.hf_tokenizer_json import HFTokenizerJson
from backend.engine.families.heartmula.bundle import (
    load_gen_config,
    mlx_weights_path,
    mlx_weights_ready,
    resolve_heartmula_bundle,
)
from backend.engine.families.heartmula.generation import estimate_hf_noise_ratio
from backend.engine.families.heartmula.generation import FRAME_RATE, SAMPLE_RATE
from backend.engine.families.heartmula.codec_mlx import HeartCodec, HeartCodecConfig
from backend.engine.families.heartmula.mula_mlx import HeartMuLa, HeartMuLaConfig
from backend.engine.families.heartmula.weights_mlx import load_mlx_weights_into_module

logger = logging.getLogger(__name__)

CODEC_CODEBOOK_SIZE = 8192  # RVQ indices 0..8191
CODEC_INVALID_TOKEN = 8192  # LM specials >= this must not be clipped into 8191


def _collect_code_diagnostics(codes_np: np.ndarray, valid_len: int) -> list[str]:
    """Collect LM code health diagnostics for long-sequence collapse triage.

    Metrics:
    - frame_repeat_ratio: fraction of code positions equal to previous frame.
    - avg_unique_per_codebook: average unique token count per codebook.
    - first/second-half stats to detect post-120s degeneration.
    """
    lines: list[str] = []
    if valid_len < 2:
        return lines
    codes = np.asarray(codes_np[:valid_len], dtype=np.int32)
    if codes.ndim != 2:
        return lines
    repeat_ratio = float(np.mean(codes[1:] == codes[:-1]))
    nq = int(codes.shape[1])

    def _slice_stats(name: str, sl: np.ndarray) -> None:
        if sl.shape[0] <= 1:
            return
        uniq_counts = [int(np.unique(sl[:, k]).size) for k in range(nq)]
        avg_unique = float(np.mean(uniq_counts))
        local_repeat = float(np.mean(sl[1:] == sl[:-1]))
        lines.append(
            "HeartMuLa LM code stats "
            f"({name}): frames={int(sl.shape[0])}, repeat={local_repeat:.3f}, "
            f"avg_unique/codebook={avg_unique:.1f}",
        )

    lines.append(
        "HeartMuLa LM code stats "
        f"(all): frames={valid_len}, repeat={repeat_ratio:.3f}",
    )
    split = int(120 * FRAME_RATE)
    _slice_stats("first_120s", codes[: min(valid_len, split)])
    if valid_len > split:
        _slice_stats("after_120s", codes[split:valid_len])
    return lines


def _lm_eval_interval(max_frames: int) -> int:
    # Quality-first: keep per-frame eval for stable autoregressive sampling under MLX lazy execution.
    # Sparse eval (every N frames) can accumulate long deferred graphs and drift token quality.
    _ = max_frames
    return 1


def _lm_progress_stride(max_frames: int) -> int:
    return 25 if max_frames <= 500 else 50


def _apply_mlx_memory_budget() -> None:
    raw = (os.environ.get("MLX_METAL_MEMORY_LIMIT") or "").strip()
    try:
        gb = int(float(raw)) if raw else 32
    except ValueError:
        gb = 32
    gb = max(16, min(gb, 128))
    try:
        mx.set_memory_limit(gb * 1024**3)
    except Exception:
        pass


class HeartMulaMlxGenerator:
    """MLX HeartMuLa + HeartCodec — weights from download-time MLX conversion."""

    def __init__(self, ctx: Any, bundle_root: Path):
        self._ctx = ctx
        self._bundle_root = Path(bundle_root)
        self._mula: HeartMuLa | None = None
        self._codec: HeartCodec | None = None
        self._tokenizer: HFTokenizerJson | None = None
        self._gen_cfg: dict | None = None
        self._mula_paths: Any = None
        self._tags_encode_cache: dict[str, List[int]] = {}
        self.last_frame_count: int = 0
        self.last_eos_early: bool = False
        self.last_hf_noise_ratio: float = 0.0
        self.last_codec_decode_mode: str = "single"
        self.last_code_diagnostics: list[str] = []
        self.last_codec_chunk_diagnostics: list[dict[str, float]] = []

    def load(self) -> None:
        paths = resolve_heartmula_bundle(self._bundle_root)
        if not mlx_weights_ready(self._bundle_root):
            raise RuntimeError(
                "HeartMuLa MLX weights missing. Install or re-install the model from the "
                f"download center so install_hooks can convert weights once ({paths.root})"
            )

        lm_dtype = mx.bfloat16
        codec_dtype = mx.float32
        logger.info("Loading HeartMuLa MLX from %s", paths.root)

        mula_cfg = HeartMuLaConfig.from_pretrained(paths.mula_torch)
        self._mula = HeartMuLa(mula_cfg)
        load_mlx_weights_into_module(
            self._mula,
            mlx_weights_path(paths.mula_torch),
            dtype=lm_dtype,
            eval_fn=self._ctx.eval,
            array_fn=self._ctx.array,
        )

        codec_cfg = HeartCodecConfig.from_pretrained(paths.codec_torch)
        self._codec = HeartCodec(codec_cfg)
        load_mlx_weights_into_module(
            self._codec,
            mlx_weights_path(paths.codec_torch),
            dtype=codec_dtype,
            eval_fn=self._ctx.eval,
            array_fn=self._ctx.array,
        )

        self._tokenizer = HFTokenizerJson.from_directory(paths.root)
        self._gen_cfg = load_gen_config(paths.gen_config)
        self._mula_paths = paths
        self._ctx.eval(self._mula.parameters(), self._codec.parameters())
        logger.info("HeartMuLa MLX ready")

    def _cfg(self) -> dict:
        if self._gen_cfg is None:
            raise RuntimeError("HeartMuLa MLX generator not loaded")
        return self._gen_cfg

    def _ensure_mula_loaded(self) -> HeartMuLa:
        if self._mula is not None:
            return self._mula
        if self._mula_paths is None:
            raise RuntimeError("HeartMuLa MLX generator not loaded")
        dtype = mx.bfloat16
        paths = self._mula_paths
        mula_cfg = HeartMuLaConfig.from_pretrained(paths.mula_torch)
        self._mula = HeartMuLa(mula_cfg)
        load_mlx_weights_into_module(
            self._mula,
            mlx_weights_path(paths.mula_torch),
            dtype=dtype,
            eval_fn=self._ctx.eval,
            array_fn=self._ctx.array,
        )
        logger.info("HeartMuLa LM re-loaded after codec phase")
        return self._mula

    def _unload_mula(self) -> None:
        if self._mula is None:
            return
        self._mula.reset_caches()
        self._mula = None
        self._ctx.clear_cache()
        gc.collect()

    def _encode_tags(self, tags_s: str, text_bos: int, text_eos: int) -> List[int]:
        if not tags_s:
            return []
        cached = self._tags_encode_cache.get(tags_s)
        if cached is not None:
            return list(cached)
        tags_ids = self._tokenizer.encode(tags_s)
        if tags_ids[0] != text_bos:
            tags_ids = [text_bos] + tags_ids
        if tags_ids[-1] != text_eos:
            tags_ids = tags_ids + [text_eos]
        self._tags_encode_cache[tags_s] = list(tags_ids)
        return tags_ids

    @staticmethod
    def _crossfade_concat(prev: np.ndarray, cur: np.ndarray, fade_samples: int) -> np.ndarray:
        if prev.size == 0:
            return cur
        if cur.size == 0:
            return prev
        ov = max(0, min(int(fade_samples), int(prev.shape[0]), int(cur.shape[0])))
        if ov <= 0:
            return np.concatenate([prev, cur], axis=0)
        t = np.linspace(0.0, 1.0, ov, dtype=np.float32)
        mixed = prev[-ov:] * (1.0 - t) + cur[:ov] * t
        return np.concatenate([prev[:-ov], mixed, cur[ov:]], axis=0)

    def _generate_waveform_segmented(
        self,
        *,
        tags: str,
        lyrics: str,
        duration: float,
        temperature: float,
        topk: int,
        cfg_scale: float,
        codec_steps: int,
        codec_guidance: float,
        long_form_temperature: float,
        long_form_topk: int,
        seed: int,
        progress_callback: Optional[Callable[[int, int], None]],
    ) -> np.ndarray:
        seg_max_sec = 120.0
        fade_sec = 2.0
        total_target = max(1, int(round(float(duration) * SAMPLE_RATE)))
        fade_samples = int(round(fade_sec * SAMPLE_RATE))
        stitched = np.zeros((0,), dtype=np.float32)
        remaining = float(duration)
        seg_idx = 0
        agg_frames = 0
        any_eos = False
        logger.info(
            "HeartMuLa segmented long-form enabled: duration=%.1fs, seg_max=%.1fs, fade=%.1fs",
            duration,
            seg_max_sec,
            fade_sec,
        )
        while remaining > 1e-6:
            if seg_idx == 0:
                seg_dur = min(seg_max_sec, remaining)
                stitched_contrib = seg_dur
            else:
                seg_dur = min(seg_max_sec, remaining + fade_sec)
                stitched_contrib = max(0.0, seg_dur - fade_sec)
            seg_seed = int(seed) + seg_idx * 10007
            logger.info(
                "HeartMuLa segmented pass %d: seg_dur=%.1fs, contrib=%.1fs, seed=%d",
                seg_idx + 1,
                seg_dur,
                stitched_contrib,
                seg_seed,
            )
            try:
                cur = self.generate_waveform(
                    tags=tags,
                    lyrics=lyrics,
                    duration=seg_dur,
                    temperature=temperature,
                    topk=topk,
                    cfg_scale=cfg_scale,
                    codec_steps=codec_steps,
                    codec_guidance=codec_guidance,
                    long_form_temperature=long_form_temperature,
                    long_form_topk=long_form_topk,
                    seed=seg_seed,
                    progress_callback=progress_callback,
                    _allow_segmented=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"HeartMuLa segmented pass {seg_idx + 1} failed: {exc}"
                ) from exc
            stitched = self._crossfade_concat(stitched, np.asarray(cur, dtype=np.float32), fade_samples)
            agg_frames += int(getattr(self, "last_frame_count", 0))
            any_eos = any_eos or bool(getattr(self, "last_eos_early", False))
            remaining -= stitched_contrib
            seg_idx += 1
            if seg_idx >= 16:
                logger.warning("HeartMuLa segmented mode hit safety segment cap (16)")
                break
        if stitched.shape[0] > total_target:
            stitched = stitched[:total_target]
        elif stitched.shape[0] < total_target:
            pad = np.zeros((total_target - stitched.shape[0],), dtype=np.float32)
            stitched = np.concatenate([stitched, pad], axis=0)
        self.last_frame_count = agg_frames
        self.last_eos_early = any_eos
        self.last_codec_decode_mode = "segmented"
        self.last_hf_noise_ratio = estimate_hf_noise_ratio(stitched, SAMPLE_RATE)
        logger.info(
            "HeartMuLa segmented done: segments=%d, stitched=%.1fs, hf_noise=%.3f",
            seg_idx,
            stitched.shape[0] / float(SAMPLE_RATE),
            self.last_hf_noise_ratio,
        )
        return stitched

    def generate_waveform(
        self,
        *,
        tags: str,
        lyrics: str,
        duration: float,
        temperature: float = 1.0,
        topk: int = 50,
        cfg_scale: float = 1.5,
        codec_steps: int = 20,
        codec_guidance: float = 1.25,
        long_form_temperature: float = 1.04,
        long_form_topk: int = 60,
        seed: int = 0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        _allow_segmented: bool = True,
    ) -> np.ndarray:
        if _allow_segmented and float(duration) > 120.0:
            return self._generate_waveform_segmented(
                tags=tags,
                lyrics=lyrics,
                duration=float(duration),
                temperature=temperature,
                topk=topk,
                cfg_scale=cfg_scale,
                codec_steps=codec_steps,
                codec_guidance=codec_guidance,
                long_form_temperature=long_form_temperature,
                long_form_topk=long_form_topk,
                seed=seed,
                progress_callback=progress_callback,
            )
        if self._codec is None or self._tokenizer is None:
            raise RuntimeError("HeartMuLa MLX generator not loaded")
        self.last_code_diagnostics = []
        self.last_codec_chunk_diagnostics = []
        mula = self._ensure_mula_loaded()
        _apply_mlx_memory_budget()
        set_random_seed(None, seed)

        cfg = self._cfg()
        text_bos = int(cfg.get("text_bos_id", 128000))
        text_eos = int(cfg.get("text_eos_id", 128001))
        audio_eos = int(cfg.get("audio_eos_id", 8193))
        empty_id = int(cfg.get("empty_id", 0))
        num_codebooks = int(mula.num_codebooks)
        parallel = num_codebooks + 1

        tags_s = (tags or "").strip().lower()
        if tags_s and not tags_s.startswith("<tag>"):
            tags_s = f"<tag>{tags_s}"
        if tags_s and not tags_s.endswith("</tag>"):
            tags_s = f"{tags_s}</tag>"

        tags_ids = self._encode_tags(tags_s, text_bos, text_eos)

        lyrics_s = (lyrics or "").strip().lower()
        lyrics_ids: List[int] = []
        if lyrics_s:
            lyrics_ids = self._tokenizer.encode(lyrics_s)
            if lyrics_ids[0] != text_bos:
                lyrics_ids = [text_bos] + lyrics_ids
            if lyrics_ids[-1] != text_eos:
                lyrics_ids = lyrics_ids + [text_eos]

        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)
        muq_idx = len(tags_ids)

        prompt_tokens = np.zeros((prompt_len, parallel), dtype=np.int32)
        if tags_ids:
            prompt_tokens[: len(tags_ids), -1] = tags_ids
        if lyrics_ids:
            prompt_tokens[len(tags_ids) + 1 :, -1] = lyrics_ids

        prompt_mask = np.zeros((prompt_len, parallel), dtype=bool)
        prompt_mask[:, -1] = True

        bs = 2 if cfg_scale != 1.0 else 1

        tokens = self._ctx.array(prompt_tokens)[None, ...]
        if cfg_scale != 1.0:
            tokens = mx.concatenate([tokens, tokens], axis=0)
        tokens_mask = self._ctx.array(prompt_mask)[None, ...]
        if cfg_scale != 1.0:
            tokens_mask = mx.concatenate([tokens_mask, tokens_mask], axis=0)

        muq_embed = mx.zeros((bs, mula.config.muq_dim), dtype=mx.float32)
        pos = mx.arange(prompt_len, dtype=mx.int32)
        if bs == 2:
            pos = mx.broadcast_to(pos[None, :], (2, prompt_len))
        else:
            pos = pos[None, :]

        max_frames = max(1, int(duration * FRAME_RATE))
        max_seq_len = prompt_len + max_frames + 8
        lm_eval_interval = _lm_eval_interval(max_frames)
        progress_stride = _lm_progress_stride(max_frames)
        self.last_eos_early = False
        mula.setup_caches(bs, max_seq_len)

        padded_buf = mx.full((bs, 1, parallel), empty_id, dtype=mx.int32)
        pad_mask_buf = mx.zeros((bs, 1, parallel), dtype=mx.bool_)
        pad_mask_buf[:, :, :num_codebooks] = True
        codes_buf = mx.zeros((max_frames, num_codebooks), dtype=mx.int32)

        def _report_progress(n_frames: int, *, force: bool = False) -> None:
            if progress_callback is None:
                return
            if force or n_frames <= 1 or n_frames % progress_stride == 0 or n_frames >= max_frames:
                progress_callback(n_frames, max_frames)

        logger.info(
            "HeartMuLa LM: up to %d frames (~%.1fs), prompt_len=%d, cfg=%.2f, topk=%d, eval_every=%d",
            max_frames,
            duration,
            prompt_len,
            cfg_scale,
            topk,
            lm_eval_interval,
        )

        t_prefill = time.monotonic()
        curr = mula.generate_frame(
            tokens=tokens,
            tokens_mask=tokens_mask,
            input_pos=pos,
            temperature=temperature,
            topk=topk,
            cfg_scale=cfg_scale,
            continuous_segments=muq_embed,
            starts=[muq_idx] * bs,
        )
        self._ctx.eval(curr)
        codes_buf[0] = curr[0]
        n_frames = 1
        _report_progress(1, force=True)
        logger.info("HeartMuLa prefill done in %.1fs", time.monotonic() - t_prefill)

        t_loop = time.monotonic()
        base_pos = int(np.asarray(pos[0, -1]))
        long_form_start = int(120 * FRAME_RATE)
        long_form_temp = max(float(temperature), float(long_form_temperature))
        long_form_topk = max(int(topk), int(long_form_topk))
        long_form_boost_logged = False
        for i in range(max_frames - 1):
            t_frame = time.monotonic()
            padded_buf[:, 0, :num_codebooks] = curr
            padded_buf[:, 0, -1] = empty_id
            next_pos = self._ctx.array([[base_pos + i + 1]] * bs, dtype=mx.int32)
            step_index = i + 2  # includes prefill frame as step 1
            if max_frames > long_form_start and step_index >= long_form_start:
                step_temp = long_form_temp
                step_topk = long_form_topk
                if not long_form_boost_logged:
                    logger.info(
                        "HeartMuLa long-form stabilize: from frame %d use temperature=%.2f topk=%d",
                        long_form_start,
                        step_temp,
                        step_topk,
                    )
                    long_form_boost_logged = True
            else:
                step_temp = temperature
                step_topk = topk
            curr = mula.generate_frame(
                tokens=padded_buf,
                tokens_mask=pad_mask_buf,
                input_pos=next_pos,
                temperature=step_temp,
                topk=step_topk,
                cfg_scale=cfg_scale,
                continuous_segments=None,
                starts=None,
            )
            if (i + 1) % lm_eval_interval == 0:
                self._ctx.eval(curr)
            frame_s = time.monotonic() - t_frame
            if frame_s >= 8.0 or (i + 1) % 10 == 0:
                logger.info(
                    "HeartMuLa LM frame %d/%d (%.2fs this frame, %.1fs elapsed)",
                    i + 2,
                    max_frames,
                    frame_s,
                    time.monotonic() - t_loop,
                )
            if mx.any(curr[0] >= audio_eos):
                self.last_eos_early = True
                break
            codes_buf[n_frames] = curr[0]
            n_frames += 1
            _report_progress(n_frames)

        mula.reset_caches()
        self.last_frame_count = n_frames
        _report_progress(n_frames, force=True)
        logger.info(
            "HeartMuLa LM done: %d frames (~%.1fs), eos_early=%s; Codec decode (%d steps)",
            n_frames,
            n_frames / FRAME_RATE,
            self.last_eos_early,
            codec_steps,
        )

        codes_np = np.asarray(codes_buf[:n_frames], dtype=np.int32)
        valid_len = n_frames
        for t in range(n_frames):
            if np.any(codes_np[t] >= CODEC_CODEBOOK_SIZE):
                valid_len = t
                break
        invalid = int(np.sum(codes_np[:valid_len] >= CODEC_CODEBOOK_SIZE))
        if invalid:
            logger.warning(
                "HeartMuLa LM emitted %d special/invalid code(s) (>= %d) before trim",
                invalid,
                CODEC_CODEBOOK_SIZE,
            )
        if valid_len == 0:
            raise RuntimeError("HeartMuLa produced no valid audio codes before EOS/special tokens")
        if valid_len < n_frames:
            self.last_eos_early = True
            logger.info(
                "HeartMuLa codec input trimmed to %d/%d frames (dropped LM special tokens)",
                valid_len,
                n_frames,
            )
        self.last_code_diagnostics = _collect_code_diagnostics(codes_np, valid_len)
        for line in self.last_code_diagnostics:
            logger.info(line)
        codes_batch = self._ctx.array(codes_np[:valid_len][None, :, :], dtype=mx.int32)

        self._unload_mula()

        # Codec flow-matching uses fresh Gaussian noise; re-seed after LM sampling.
        set_random_seed(None, int(seed) + 1_000_003)

        codec_frame_rate = float(self._codec.config.frame_rate)
        codec_duration = valid_len / codec_frame_rate
        audio = self._codec.detokenize(
            codes=codes_batch,
            duration=codec_duration,
            num_steps=codec_steps,
            guidance_scale=codec_guidance,
        )
        self.last_codec_decode_mode = getattr(self._codec, "last_detokenize_mode", "single")
        self.last_codec_chunk_diagnostics = list(
            getattr(self._codec, "last_chunk_diagnostics", []) or []
        )
        logger.info(
            "HeartMuLa Codec done: mode=%s, frames=%d, ~%.1fs",
            self.last_codec_decode_mode,
            valid_len,
            codec_duration,
        )
        self._ctx.eval(audio)

        wf = np.array(audio.astype(mx.float32)).flatten()
        wf = wf - float(wf.mean())
        peak = float(np.abs(wf).max())
        if peak < 1e-8:
            raise RuntimeError("HeartMuLa decode produced near-silent audio")
        if peak > 1.0:
            wf = (wf / peak * 0.99).astype(np.float32)
        else:
            wf = wf.astype(np.float32)
        self.last_hf_noise_ratio = estimate_hf_noise_ratio(wf, SAMPLE_RATE)
        return wf
