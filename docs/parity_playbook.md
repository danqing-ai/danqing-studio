# Parity Playbook (SOP)

This playbook captures the regression-debug workflow used to recover mflux parity for Flux2 and Z-Image families.

## Goal Hierarchy

1. **Product usable first**: recover runnable baseline and avoid generation paralysis.
2. **Case pass second**: benchmark status (`PASS/WARN/FAIL`) must improve case-by-case.
3. **Pixel parity third**: pursue PSNR alignment only after semantic contracts are verified.

## Standard Workflow

### 1) Confirm Failure Type

- Run a single failing case first: `make bench-mflux-case ID=<case-id>`.
- Classify the failure:
  - `runtime` (crash / missing weights / key mismatch)
  - `semantic` (same inputs, different pipeline behavior)
  - `numeric` (same semantics, different tensor math)

Do not start low-level math diffing before semantic checks are complete.

### 2) Lock Inputs and Invariants

- Freeze benchmark inputs: prompt, seed, size, steps, guidance, scheduler.
- Record invariants in logs:
  - scheduler name and step count
  - `requires_sigma_shift`, `use_empirical_mu`, `scheduler_mu`
  - initial latent shape and dtype
  - whether CFG is enabled, and negative prompt path used

If invariants are not logged, add logs first.

### 3) Check Contract Layer Before Model Math

Verify these runtime contracts before touching transformer internals:

- **Scheduler semantics**: registry override vs config fallback behavior.
- **Guidance semantics**: `supports_guidance` handling and CFG dual-pass gate.
- **Noise semantics**: initial latent layout and sampling dtype path.
- **Text semantics**: encoder choice and negative prompt eligibility.
- **Tokenizer template semantics**: chat-template kwargs (for example `enable_thinking`) must match reference runtime defaults.

Most regressions in this cycle were contract-layer regressions, not model architecture errors.

### 4) Isolate by Stage

Use stage isolation to cut search space:

1. text encoder output parity
2. scheduler timetable/sigma parity
3. initial noise parity
4. transformer forward parity (single step)
5. VAE decode parity

Only move to next stage after current stage is bounded.

### 5) Fix With “Entry Consolidation”

- Prefer centralized semantic entrypoints instead of duplicating inline branches.
- Keep behavior unchanged first; refactor shape only.
- Add contract tests with each fix to prevent regressions.

### 6) Re-Run and Tighten

- Re-run affected case first.
- Re-run family set (e.g. Flux2 + Z-Image create/rewrite).
- Promote from `WARN` to `PASS` only after repeated stable results.

## Regression Checklist (Quick)

- [ ] registry flag fallback does not override config with `None`
- [ ] `requires_sigma_shift` and `use_empirical_mu` are resolved deterministically
- [ ] z-image noise layout uses mflux-compatible convention
- [ ] flux2 noise sampling uses fp32 sampling then cast when needed
- [ ] z-image family (including turbo) uses mflux-aligned tokenizer chat-template settings (`enable_thinking`)
- [ ] structured-prompt families do not enter negative-prompt JSON path
- [ ] benchmark contract tests cover fallback + override semantics

## Known High-Impact Pattern (2026-05)

- **Symptom**: only one or two parity cases remain `WARN` while neighboring cases are already `PASS`.
- **Root cause in this cycle**: tokenizer chat-template mismatch (`enable_thinking`) for `z-image-turbo`, causing different token sequence despite identical prompt/seed/steps.
- **Fast check**: compare token ids + attention mask lengths between DanQing and mflux before touching transformer math.
- **Fix principle**: align registry/runtime tokenizer kwargs to reference first, then re-run targeted case and full family subset.

## Anti-Patterns

- Jumping directly into RoPE/attention math when scheduler/noise semantics are unverified.
- Refactor-only dedup without contract tests.
- Accepting hidden fallback behavior in generation path (violates fail-loud principle).

## Deliverables Required Per Parity Fix

1. one benchmark evidence run (case-level)
2. one contract test for the fixed semantic
3. one short note in this playbook if a new regression pattern is discovered
