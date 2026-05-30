# HeartMuLa codec parity fixtures

Gold files for `make bench-audio-sanity-heartmula` (Codec-only gate runs after E2E sanity).

## Files

| File | Role |
|------|------|
| `codec_parity_manifest.json` | Case params (codec steps, seed, filenames) |
| `codec_parity_10s_codes.npy` | Fixed LM output `(T, 8)` int32, values `0..8191` |
| `codec_parity_10s_reference.wav` | heartlib-mlx `HeartCodec.detokenize(codes)` — gold waveform |

Until `codes.npy` and `reference.wav` exist, the benchmark **SKIP**s (exit 0).

## One-time setup

### 1) Export codes from DanQing (LM only path)

With HeartMuLa bundle installed under `models/Audio/heartmula-oss-3b-happy-new-year`:

```bash
python scripts/export_heartmula_codec_parity_fixtures.py generate-codes \
  --output-dir tests/benchmark/fixtures/heartmula \
  --duration 10 --seed 42
```

This sets `DANQING_HEARTMULA_DUMP_CODES` during generation and writes `codec_parity_10s_codes.npy`.

### 2) Decode the same codes with heartlib-mlx (reference WAV)

Clone [heartlib-mlx](https://github.com/Acelogic/heartlib-mlx), install deps, point `--heartlib-repo` at the clone, then:

```bash
python scripts/export_heartmula_codec_parity_fixtures.py decode-heartlib \
  --output-dir tests/benchmark/fixtures/heartmula \
  --heartlib-repo /path/to/heartlib-mlx
```

Or decode manually in heartlib and copy the WAV to `codec_parity_10s_reference.wav` (48 kHz mono, same `codec_steps` / `codec_guidance` / `codec_seed` as manifest).

### 3) Run parity

Included in HeartMuLa sanity (no separate make target):

```bash
make bench-audio-sanity-heartmula
# or
make bench-sanity-case ID=heartmula-oss-3b-happy-new-year-sanity
```

After E2E generation checks, the runner decodes fixed `codes.npy` with DanQing HeartCodec and compares to `reference.wav` (SI-SDR + correlation).

## Metrics

Default gates (`tests/benchmark/heartmula_codec_parity.py`):

- **PASS**: SI-SDR ≥ 18 dB and correlation ≥ 0.90
- **WARN**: SI-SDR ≥ 12 dB
- **FAIL**: below WARN

If output is pure noise, SI-SDR will be strongly negative — that confirms a Codec/weights bug, not LM sampling.

## Local debug (no reference yet)

```bash
python scripts/export_heartmula_codec_parity_fixtures.py decode-danqing \
  --codes tests/benchmark/fixtures/heartmula/codec_parity_10s_codes.npy \
  --output /tmp/dq_codec.wav
```

Listen to `/tmp/dq_codec.wav` before investing in heartlib reference export.
