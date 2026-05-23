"""HeartCodec - Neural Audio Codec with Flow Matching Decoder."""

from __future__ import annotations

import math
from typing import List, Optional, Union
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# heartlib ``detokenize()`` uses this default for hop/chunk sizing (not full song length).
_HEARTLIB_CHUNK_DURATION_SEC = 29.76

from backend.engine.families.heartmula.mlx.heartcodec.configuration import HeartCodecConfig
from backend.engine.families.heartmula.mlx.heartcodec.scalar_codec import ScalarModel
from backend.engine.families.heartmula.mlx.heartcodec.flow_matching import FlowMatchingDecoder


class HeartCodec(nn.Module):
    """HeartCodec: Neural Audio Codec with Flow Matching Decoder.

    HeartCodec is a 12.5Hz neural audio codec that combines:
    1. ScalarModel: Convolutional encoder/decoder for audio
    2. FlowMatchingDecoder: Transformer-based generative decoder

    The codec operates at 48kHz sample rate with a frame rate of
    12.5Hz (3840 samples per frame).

    Args:
        config: HeartCodecConfig with model hyperparameters.
    """

    def __init__(self, config: HeartCodecConfig):
        super().__init__()
        self.config = config

        # Scalar codec for audio encoding/decoding
        self.scalar_model = ScalarModel(
            num_bands=config.num_bands,
            sample_rate=config.sample_rate,
            causal=config.causal,
            num_samples=config.num_samples,
            downsample_factors=config.downsample_factors,
            downsample_kernel_sizes=config.downsample_kernel_sizes,
            upsample_factors=config.upsample_factors,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            latent_hidden_dim=config.latent_hidden_dim,
            default_kernel_size=config.default_kernel_size,
            delay_kernel_size=config.delay_kernel_size,
            init_channel=config.init_channel,
            res_kernel_size=config.res_kernel_size,
        )

        # Flow matching decoder for high-quality synthesis
        self.flow_matching = FlowMatchingDecoder(
            dim=config.dim,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            num_quantizers=config.num_quantizers,
            attention_head_dim=config.attention_head_dim,
            in_channels=config.in_channels,
            num_attention_heads=config.num_attention_heads,
            num_layers=config.num_layers,
            num_layers_2=config.num_layers_2,
            out_channels=config.out_channels,
            use_cosine_sim=config.use_cosine_sim,
            decay=config.decay,
            commitment_weight=config.commitment_weight,
            threshold_ema_dead_code=config.threshold_ema_dead_code,
        )

    def encode(self, audio: mx.array) -> mx.array:
        """Encode audio to quantized latent representation.

        Args:
            audio: Audio waveform of shape (batch, samples) or (batch, samples, 1).

        Returns:
            Quantized latent of shape (batch, frames, latent_dim).
        """
        return self.scalar_model.encode(audio)

    def decode(self, latent: mx.array) -> mx.array:
        """Decode quantized latent to audio waveform.

        Args:
            latent: Quantized latent of shape (batch, frames, latent_dim).

        Returns:
            Audio waveform of shape (batch, samples, 1).
        """
        return self.scalar_model.decode(latent)

    def _normalize_codes_layout(self, codes: mx.array) -> mx.array:
        """Accept ``(batch, time, K)`` or heartlib ``(batch, K, time)`` layout."""
        if codes.ndim != 3:
            raise RuntimeError(
                f"HeartCodec codes must be rank-3, got shape {tuple(codes.shape)}"
            )
        nq = int(self.config.num_quantizers)
        _, d1, d2 = codes.shape
        if d2 == nq:
            return codes
        if d1 == nq and d2 != nq:
            return codes.transpose(0, 2, 1)
        raise RuntimeError(
            f"HeartCodec codes layout unrecognized: shape={tuple(codes.shape)}, "
            f"expected (batch, time, {nq}) or (batch, {nq}, time)"
        )

    def _latent_to_waveform_chunk(self, latent: mx.array) -> mx.array:
        """Scalar decode one overlap chunk; returns ``(samples,)`` mono."""
        bsz, t, f = latent.shape
        latent = latent.reshape(bsz, t, 2, f // 2)
        latent = latent.transpose(0, 2, 1, 3)
        latent = latent.reshape(bsz * 2, t, f // 2)
        audio = self.scalar_model.decode(latent)
        samples = int(audio.shape[1])
        audio = audio.reshape(bsz, 2, samples, 1)
        audio = mx.mean(audio, axis=1)
        return audio[0, :, 0]

    def detokenize(
        self,
        codes: mx.array,
        duration: float = _HEARTLIB_CHUNK_DURATION_SEC,
        num_steps: int = 10,
        guidance_scale: float = 1.25,
    ) -> mx.array:
        """Convert discrete codes to waveform (heartlib chunked overlap-add).

        Args:
            codes: ``(batch, time, K)`` or ``(batch, K, time)``.
            duration: Chunk template seconds for hop sizing (default 29.76, per heartlib).
                Output length is derived from code frame count, not this value.
            num_steps: Flow-matching ODE steps.
            guidance_scale: Codec CFG scale.

        Returns:
            Mono waveform ``(batch, samples, 1)``.
        """
        codes = self._normalize_codes_layout(codes)
        batch_size = codes.shape[0]
        nq = int(codes.shape[2])
        content_frames = int(codes.shape[1])
        frame_rate = float(self.config.frame_rate)
        sample_rate = int(self.config.sample_rate)

        target_samples = int(content_frames / frame_rate * sample_rate)
        chunk_duration = float(duration)
        min_samples = int(chunk_duration * frame_rate)
        hop_samples = min_samples // 93 * 80
        ovlp_samples = min_samples - hop_samples
        ovlp_frames = ovlp_samples * 2
        latent_length = int(chunk_duration * 25)

        def _pad_codes_time(target_len: int) -> None:
            nonlocal codes
            cur = int(codes.shape[1])
            if cur >= target_len:
                if cur > target_len:
                    codes = codes[:, :target_len, :]
                return
            if cur == 0:
                pad = mx.zeros((batch_size, target_len - cur, nq), dtype=codes.dtype)
            else:
                last = codes[:, -1:, :]
                pad = mx.broadcast_to(last, (batch_size, target_len - cur, nq))
            codes = mx.concatenate([codes, pad], axis=1)

        first_latent = mx.random.normal(
            shape=(batch_size, latent_length, self.flow_matching.out_channels)
        )
        first_latent_length = 0

        if content_frames < min_samples:
            _pad_codes_time(min_samples)

        if (content_frames - ovlp_frames) % hop_samples > 0:
            len_codes = (
                math.ceil((content_frames - ovlp_samples) / float(hop_samples))
                * hop_samples
                + ovlp_samples
            )
            _pad_codes_time(len_codes)

        codes_len = int(codes.shape[1])

        latent_list: List[mx.array] = []
        for sinx in range(0, codes_len - hop_samples + 1, hop_samples):
            codes_chunk = codes[:, sinx : sinx + min_samples, :]
            if sinx == 0 or ovlp_frames == 0:
                latents = self.flow_matching.inference_codes(
                    codes_chunk,
                    true_latents=first_latent,
                    latent_length=latent_length,
                    incontext_length=first_latent_length,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    scenario="other_seg",
                )
                latent_list.append(latents)
            else:
                prev = latent_list[-1]
                true_latent = prev[:, -ovlp_frames:, :]
                # heartlib: incontext_length is overlap width *before* padding to latent_length
                incontext_length = int(true_latent.shape[1])
                len_add = latent_length - incontext_length
                if len_add > 0:
                    true_latent = mx.concatenate(
                        [
                            true_latent,
                            mx.random.normal(
                                shape=(
                                    batch_size,
                                    len_add,
                                    self.flow_matching.out_channels,
                                ),
                                dtype=true_latent.dtype,
                            ),
                        ],
                        axis=1,
                    )
                latents = self.flow_matching.inference_codes(
                    codes_chunk,
                    true_latents=true_latent,
                    latent_length=latent_length,
                    incontext_length=incontext_length,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    scenario="other_seg",
                )
                latent_list.append(latents)

        latent_list[0] = latent_list[0][:, first_latent_length:, :]
        min_audio_samples = int(chunk_duration * sample_rate)
        hop_audio = min_audio_samples // 93 * 80
        ovlp_audio = min_audio_samples - hop_audio

        output: Optional[mx.array] = None
        for i, latent in enumerate(latent_list):
            cur = self._latent_to_waveform_chunk(latent)
            cur = cur[:min_audio_samples]
            if output is None:
                output = cur
            elif ovlp_audio == 0:
                output = mx.concatenate([output, cur], axis=0)
            else:
                t = mx.linspace(0.0, 1.0, ovlp_audio)
                ov_win = mx.concatenate([t, 1.0 - t], axis=0)
                tail = output[-ovlp_audio:] * ov_win[-ovlp_audio:]
                head = cur[:ovlp_audio] * ov_win[:ovlp_audio]
                output = mx.concatenate(
                    [output[:-ovlp_audio], tail + head, cur[ovlp_audio:]],
                    axis=0,
                )

        assert output is not None
        output = output[:target_samples]
        return output[:, None, None]

    def tokenize(self, audio: mx.array) -> mx.array:
        """Encode audio to discrete codes.

        Args:
            audio: Audio waveform of shape (batch, samples) or (batch, samples, 1).

        Returns:
            Audio codes of shape (batch, frames, num_quantizers).
        """
        # Get quantized latent
        latent = self.encode(audio)

        # Quantize to codes using flow matching's VQ
        codes = self.flow_matching.vq.encode(latent)

        return codes

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "HeartCodec":
        """Load a pretrained HeartCodec model.

        Args:
            path: Path to the model directory.
            dtype: Data type for model weights.

        Returns:
            HeartCodec instance with loaded weights.
        """
        path = Path(path)

        # Load config
        config = HeartCodecConfig.from_pretrained(path)

        # Create model
        model = cls(config)

        # Load weights using MLX's native loader (handles bfloat16 properly)
        weights_path = path / "model.safetensors"
        if weights_path.exists():
            weights = mx.load(str(weights_path))

            # Convert to target dtype if different
            weights = {k: v.astype(dtype) for k, v in weights.items()}

            # Load into model (strict=False to ignore PreProcessor/PostProcessor weights)
            model.load_weights(list(weights.items()), strict=False)
            mx.eval(model.parameters())

        return model

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save the model to a directory.

        Args:
            path: Path to save the model.
        """
        from safetensors.numpy import save_file
        import numpy as np

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save_pretrained(path)

        # Save weights
        weights = dict(self.parameters())
        # Convert to numpy for safetensors
        np_weights = {k: np.array(v) for k, v in weights.items()}
        save_file(np_weights, str(path / "model.safetensors"))

    def __call__(
        self,
        audio: Optional[mx.array] = None,
        codes: Optional[mx.array] = None,
        num_steps: int = 10,
        guidance_scale: float = 1.25,
    ) -> mx.array:
        """Forward pass for encoding or decoding.

        Args:
            audio: Input audio for encoding (optional).
            codes: Input codes for decoding (optional).
            num_steps: ODE integration steps for decoding.
            guidance_scale: CFG scale for decoding.

        Returns:
            Encoded codes (if audio provided) or decoded audio (if codes provided).
        """
        if audio is not None:
            return self.tokenize(audio)
        elif codes is not None:
            return self.detokenize(codes, num_steps=num_steps, guidance_scale=guidance_scale)
        else:
            raise ValueError("Either audio or codes must be provided")
