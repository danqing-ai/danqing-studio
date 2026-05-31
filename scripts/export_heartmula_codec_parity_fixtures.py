#!/usr/bin/env python3
"""Prepare HeartMuLa codec parity fixtures (codes + heartlib-mlx reference WAV)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_FIXTURES = PROJECT_ROOT / "tests" / "benchmark" / "fixtures" / "heartmula"
DEFAULT_CODES_NAME = "codec_parity_10s_codes.npy"
DEFAULT_REF_NAME = "codec_parity_10s_reference.wav"
DEFAULT_MANIFEST_NAME = "codec_parity_manifest.json"


def _load_manifest(fixtures_dir: Path) -> dict:
    path = fixtures_dir / DEFAULT_MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cmd_generate_codes(args: argparse.Namespace) -> int:
    fixtures = Path(args.output_dir)
    fixtures.mkdir(parents=True, exist_ok=True)
    codes_path = fixtures / DEFAULT_CODES_NAME
    manifest = _load_manifest(fixtures) if (fixtures / DEFAULT_MANIFEST_NAME).is_file() else {}

    env = os.environ.copy()
    env["DANQING_HEARTMULA_DUMP_CODES"] = str(codes_path)

    cmd = [
        str(PROJECT_ROOT / "bin" / "danqing-audio-generate"),
        "--model",
        "heartmula-oss-3b-happy-new-year",
        "--prompt",
        args.prompt,
        "--lyrics",
        args.lyrics,
        "--duration",
        str(int(args.duration)),
        "--seed",
        str(args.seed),
        "--codec-steps",
        str(int(manifest.get("codec_steps", args.codec_steps))),
        "--output",
        str(fixtures / "_dump_sidecar.wav"),
    ]
    if args.cfg_scale is not None:
        cmd.extend(["--guidance", str(args.cfg_scale)])

    print(f"[generate-codes] Running: {' '.join(cmd)}")
    print(f"[generate-codes] Will dump codes to {codes_path}")
    proc = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        print("[generate-codes] danqing-audio-generate failed", file=sys.stderr)
        return proc.returncode
    if not codes_path.is_file():
        print(f"[generate-codes] codes not written: {codes_path}", file=sys.stderr)
        return 1
    arr = __import__("numpy").load(codes_path)
    print(f"[generate-codes] OK shape={arr.shape} dtype={arr.dtype}")
    print(
        "[generate-codes] Next: decode same codes with heartlib-mlx → reference WAV:\n"
        f"  python {Path(__file__).name} decode-heartlib --output-dir {fixtures} "
        "--heartlib-repo /path/to/heartlib-mlx"
    )
    return 0


def cmd_decode_danqing(args: argparse.Namespace) -> int:
    from tests.benchmark.heartmula_codec_parity import (
        decode_codes_with_danqing_codec,
        load_codes_npy,
        resolve_heartmula_bundle_root,
    )

    codes = load_codes_npy(Path(args.codes))
    manifest = _load_manifest(Path(args.manifest_dir)) if args.manifest_dir else {}
    bundle = Path(args.bundle) if args.bundle else resolve_heartmula_bundle_root()
    wf = decode_codes_with_danqing_codec(
        bundle,
        codes,
        codec_steps=int(manifest.get("codec_steps", args.codec_steps)),
        codec_guidance=float(manifest.get("codec_guidance", args.codec_guidance)),
        chunk_duration_sec=float(manifest.get("chunk_duration_sec", 29.76)),
        codec_seed=int(manifest.get("codec_seed", args.codec_seed)),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    sr = int(manifest.get("sample_rate", 48_000))
    sf.write(str(out), wf, sr)
    print(f"[decode-danqing] wrote {out} ({wf.shape[0]} samples @ {sr} Hz)")
    return 0


def cmd_decode_heartlib(args: argparse.Namespace) -> int:
    """Decode fixed codes with heartlib-mlx; write gold reference WAV."""
    fixtures = Path(args.output_dir)
    manifest = _load_manifest(fixtures)
    codes_path = fixtures / manifest["codes_file"]
    ref_path = fixtures / manifest["reference_wav"]
    if not codes_path.is_file():
        print(f"[decode-heartlib] missing codes: {codes_path}", file=sys.stderr)
        return 1

    heartlib = Path(args.heartlib_repo).resolve()
    src = heartlib / "src"
    if not src.is_dir():
        print(f"[decode-heartlib] missing src/: {src}", file=sys.stderr)
        return 1

    codec_path = Path(args.codec_checkpoint) if args.codec_checkpoint else None
    if codec_path is None:
        from tests.benchmark.cases import HEARTMULA_AUDIO_BUNDLE, resolve_benchmark_data_root

        bundle = resolve_benchmark_data_root() / HEARTMULA_AUDIO_BUNDLE
        for name in ("HeartCodec-oss", "HeartCodec-oss-20260123"):
            candidate = bundle / name / "mlx"
            if (candidate / "model.safetensors").is_file():
                codec_path = candidate
                break
    if codec_path is None or not (codec_path / "model.safetensors").is_file():
        print(
            "[decode-heartlib] HeartCodec MLX weights not found; pass --codec-checkpoint",
            file=sys.stderr,
        )
        return 1

    sys.path.insert(0, str(src))
    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from safetensors import safe_open

    from heartlib_mlx.heartcodec.modeling import HeartCodec

    codes = np.load(codes_path)
    if codes.ndim == 3 and codes.shape[0] == 1:
        codes = codes[0]

    mx.random.seed(int(manifest.get("codec_seed", args.codec_seed)))
    codec = HeartCodec.from_pretrained(str(codec_path), dtype=mx.float32)
    # DanQing MLX bundles store TimestepEmbedder weights under ``.mlp.*``; heartlib
    # expects ``.linear_*`` — remap so parity reference uses trained weights.
    with safe_open(str(codec_path / "model.safetensors"), framework="numpy") as f:
        remap: dict[str, mx.array] = {}
        for key in f.keys():
            if ".timestep_embedder.mlp." not in key:
                continue
            nk = key.replace(".timestep_embedder.mlp.linear_1.", ".timestep_embedder.linear_1.")
            nk = nk.replace(".timestep_embedder.mlp.linear_2.", ".timestep_embedder.linear_2.")
            remap[nk] = mx.array(f.get_tensor(key))
    if remap:
        codec.load_weights(list(remap.items()), strict=False)
        mx.eval(codec.parameters())

    mx.random.seed(int(manifest.get("codec_seed", args.codec_seed)))
    duration = float(codes.shape[0]) / 12.5
    codes_mx = mx.array(codes[None, :, :], dtype=mx.int32)
    audio = codec.detokenize(
        codes_mx,
        duration=duration,
        num_steps=int(manifest.get("codec_steps", args.codec_steps)),
        guidance_scale=float(manifest.get("codec_guidance", args.codec_guidance)),
    )
    mx.eval(audio)
    wf = np.array(audio.astype(mx.float32)).reshape(-1)
    wf = wf - float(wf.mean())
    peak = float(np.abs(wf).max())
    if peak > 1e-8 and peak > 1.0:
        wf = (wf / peak * 0.99).astype(np.float32)
    elif peak > 1e-8:
        wf = wf.astype(np.float32)
    sf.write(str(ref_path), wf, int(manifest.get("sample_rate", 48_000)))
    print(f"[decode-heartlib] OK {ref_path} ({wf.shape[0]} samples, peak={peak:.4f})")
    return 0


def cmd_write_manifest(args: argparse.Namespace) -> int:
    fixtures = Path(args.output_dir)
    fixtures.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "case_id": "heartmula-codec-parity-10s",
        "description": "Fixed LM codes decoded by heartlib-mlx vs DanQing HeartCodec",
        "codes_file": DEFAULT_CODES_NAME,
        "reference_wav": DEFAULT_REF_NAME,
        "sample_rate": 48000,
        "codec_steps": args.codec_steps,
        "codec_guidance": args.codec_guidance,
        "chunk_duration_sec": 29.76,
        "codec_seed": args.codec_seed,
        "source": "heartlib-mlx",
        "source_notes": args.notes or "",
    }
    path = fixtures / DEFAULT_MANIFEST_NAME
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"[write-manifest] {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate-codes", help="Run DanQing LM+codec; dump codes via env")
    p_gen.add_argument("--output-dir", default=str(DEFAULT_FIXTURES))
    p_gen.add_argument("--prompt", default="pop, female vocal, acoustic, melodic")
    p_gen.add_argument("--lyrics", default="[verse]\nHello world\n[chorus]\nSing along")
    p_gen.add_argument("--duration", type=float, default=10.0)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--codec-steps", type=int, default=20)
    p_gen.add_argument("--cfg-scale", type=float, default=None)
    p_gen.set_defaults(func=cmd_generate_codes)

    p_dq = sub.add_parser("decode-danqing", help="DanQing HeartCodec only → WAV")
    p_dq.add_argument("--codes", required=True)
    p_dq.add_argument("--output", required=True)
    p_dq.add_argument("--manifest-dir", default=str(DEFAULT_FIXTURES))
    p_dq.add_argument("--bundle", default="")
    p_dq.add_argument("--codec-steps", type=int, default=20)
    p_dq.add_argument("--codec-guidance", type=float, default=1.25)
    p_dq.add_argument("--codec-seed", type=int, default=42424243)
    p_dq.set_defaults(func=cmd_decode_danqing)

    p_hl = sub.add_parser("decode-heartlib", help="heartlib-mlx decode → reference WAV")
    p_hl.add_argument("--output-dir", default=str(DEFAULT_FIXTURES))
    p_hl.add_argument("--heartlib-repo", required=True)
    p_hl.add_argument("--codec-checkpoint", default="")
    p_hl.add_argument("--codec-steps", type=int, default=20)
    p_hl.add_argument("--codec-guidance", type=float, default=1.25)
    p_hl.add_argument("--codec-seed", type=int, default=42424243)
    p_hl.add_argument("--strict", action="store_true", help="Exit 2 if helper script missing")
    p_hl.set_defaults(func=cmd_decode_heartlib)

    p_man = sub.add_parser("write-manifest", help="Write default manifest JSON")
    p_man.add_argument("--output-dir", default=str(DEFAULT_FIXTURES))
    p_man.add_argument("--codec-steps", type=int, default=20)
    p_man.add_argument("--codec-guidance", type=float, default=1.25)
    p_man.add_argument("--codec-seed", type=int, default=42424243)
    p_man.add_argument("--notes", default="")
    p_man.set_defaults(func=cmd_write_manifest)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
