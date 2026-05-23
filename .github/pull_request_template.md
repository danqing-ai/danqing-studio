## What / Why

- What changed:
- Why this change is needed:

## Scope

- [ ] Image
- [ ] Video
- [ ] Audio
- [ ] Frontend
- [ ] Packaging / Build
- [ ] Docs only

## New Model Integration Checklist (if applicable)

Reference: `docs/engine_new_model_checklist.md`

- [ ] Updated `default_config/models_registry.json` (+ synced runtime registry when required)
- [ ] Added/updated `backend/engine/config/model_configs.py`
- [ ] Added/updated family implementation under `backend/engine/families/<family>/`
- [ ] Registered transformer/remap/text-encoder in `backend/engine/_transformer_registry.py`
- [ ] Pipeline behavior remains registry-driven (no long-term `if family == ...` branch)
- [ ] Fail-loud behavior preserved (no silent fallback on load/inference errors)

## Reuse / Architecture Notes

- Reused `backend/engine/common/*` modules:
- Family-specific logic intentionally kept local:
- Any backend-specific hook (`*_mlx.py` / `*_cuda.py`) added and why:

## Validation

- [ ] `python -m py_compile <touched files>`
- [ ] `make verify-engine-stack` (includes governance gates + `test-engine-unit`)
- [ ] Minimal inference path validated (CLI/bench or equivalent)

Commands/results:

```text
# paste relevant commands and key outputs
```

## Risks / Follow-ups

- Risks:
- Follow-up tasks:
