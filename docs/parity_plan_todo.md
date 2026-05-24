# Parity Refactor Plan & Todo (2026-05)

## Plan

### P0 - Stabilize semantic contracts

- Keep pipeline behavior unchanged while consolidating semantic entrypoints.
- Land minimal `FamilyRuntimeContract` and `SchedulerSemanticsResolver`.
- Ensure all current passing cases remain passing.

### P1 - Prevent regression recurrence

- Add contract tests for:
  - config fallback semantics
  - registry override semantics
  - guidance/negative prompt gates
  - family-specific latent noise layout
- Wire these tests into engine unit verification path.

### P2 - Complete parity closure

- Focus on remaining non-PASS cases (currently `z-image-turbo-rewrite`).
- Use stage-isolation checklist from `docs/parity_playbook.md`.
- Only optimize performance paths after parity closure is stable.

## Todo

- [x] Write parity SOP in `docs/parity_playbook.md`.
- [x] Add minimal runtime/scheduler semantic contract skeleton.
- [x] Route `ImagePipeline.run` and `run_edit` through contract/resolver entrypoints.
- [x] Add 3-5 contract tests (runtime + scheduler semantics).
- [ ] Re-run focused benchmark: `z-image-turbo-rewrite`.
- [ ] Re-run full image parity subset: Flux2 + Z-Image + Z-Image-Turbo create/rewrite.
- [ ] If any case regresses, patch contract layer first, then add missing invariant test.
- [ ] After parity lock, evaluate whether video/audio pipelines should reuse the same resolver pattern.

## Exit Criteria

- All target parity cases reach `PASS` or have explicit temporary waiver with owner and reason.
- No silent semantic fallback in pipeline scheduler/runtime decisions.
- Contract tests fail on intentional reintroduction of this cycle's regressions.
