"""Audio preprocessing for LongCat-Video-Avatar inference."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import numpy as np


def linear_interpolate_features(
    features: mx.array,
    input_fps: float,
    output_fps: float,
    output_len: Optional[int] = None,
) -> mx.array:
    _b, t, _d = features.shape
    if output_len is None:
        output_len = int(t / input_fps * output_fps)
    if output_len == t:
        return features
    src_idx = mx.linspace(0.0, t - 1, output_len)
    lo = mx.minimum(mx.floor(src_idx).astype(mx.int32), t - 1)
    hi = mx.minimum(lo + 1, t - 1)
    frac = (src_idx - lo).astype(features.dtype)[None, :, None]
    feat_lo = features[:, lo, :]
    feat_hi = features[:, hi, :]
    return feat_lo * (1.0 - frac) + feat_hi * frac


def loudness_normalize(audio: np.ndarray, sample_rate: int = 16000, target_lufs: float = -23.0) -> np.ndarray:
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(audio)
        return pyln.normalize.loudness(audio, loudness, target_lufs)
    except (ImportError, ValueError):
        rms = float(np.sqrt(np.mean(audio**2)) + 1e-8)
        target_rms = 10 ** ((target_lufs - 0) / 20)
        return audio * (target_rms / rms)


def group_pool_whisper_hidden_states(hidden_states: list[mx.array]) -> mx.array:
    if len(hidden_states) != 33:
        raise RuntimeError(f"Whisper hidden states: expected 33 layers, got {len(hidden_states)}")
    feats = []
    for start, end in [(0, 8), (8, 16), (16, 24), (24, 32)]:
        stacked = mx.stack(hidden_states[start:end], axis=0)
        feats.append(mx.mean(stacked, axis=0))
    feats.append(hidden_states[32])
    return mx.stack(feats, axis=2)


def whisper_encode_audio_to_groups(
    whisper_encoder,
    mel_features: mx.array,
    enc_chunk: int = 3000,
) -> mx.array:
    _b, _, t_mel = mel_features.shape
    out_chunks: list[mx.array] = []
    for start in range(0, t_mel, enc_chunk):
        chunk = mel_features[:, :, start : start + enc_chunk]
        hidden_states = whisper_encoder(chunk, return_all_hidden_states=True)
        out_chunks.append(group_pool_whisper_hidden_states(hidden_states))
    return mx.concatenate(out_chunks, axis=1) if len(out_chunks) > 1 else out_chunks[0]


def build_avatar_audio_embeddings(
    audio_groups: mx.array,
    fps: int = 25,
    enc_fps: int = 50,
    audio_window: int = 5,
) -> mx.array:
    b, t_enc, g, d = audio_groups.shape
    flat = audio_groups.reshape(b, t_enc, g * d)
    target_len = int(t_enc / enc_fps * fps)
    resampled = linear_interpolate_features(
        flat, input_fps=float(enc_fps), output_fps=float(fps), output_len=target_len
    )
    per_frame = resampled.reshape(b, target_len, g, d)
    half = audio_window // 2
    if half > 0:
        first = per_frame[:, :1]
        last = per_frame[:, -1:]
        left_pad = mx.broadcast_to(first, (b, half, g, d))
        right_pad = mx.broadcast_to(last, (b, half, g, d))
        padded = mx.concatenate([left_pad, per_frame, right_pad], axis=1)
    else:
        padded = per_frame
    windows = []
    for offset in range(audio_window):
        windows.append(padded[:, offset : offset + target_len])
    return mx.stack(windows, axis=2)


def load_audio_mel(audio_path: str, *, sample_rate: int = 16000) -> mx.array:
    try:
        import librosa
    except ImportError as e:
        raise RuntimeError("librosa is required for avatar audio — pip install librosa") from e
    from transformers import WhisperFeatureExtractor

    audio, _sr = librosa.load(str(audio_path), sr=sample_rate)
    audio = loudness_normalize(audio.astype(np.float32), sample_rate=sample_rate)
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    inputs = fe(audio, sampling_rate=sample_rate, return_tensors="np")
    mel = inputs.input_features
    return mx.array(mel.astype(np.float32))
