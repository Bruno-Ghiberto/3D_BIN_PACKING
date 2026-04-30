<!-- Pull-request template (T044, Constitution §Pull-request workflow). -->

## Summary

<!--
One or two sentences. What does this PR change, and why?
-->

## Linked spec / issue

<!--
Reference the SpecKit feature, ADR, or issue this PR delivers against.
Examples:
- specs/001-public-release-hardening/tasks.md (T0XX..T0YY)
- Closes #123
- ADR-0008
-->

## Changes

<!--
Bullet list of the meaningful changes. Group by area when helpful
(library / CLI / docs / CI). Mention removed code as well as added.
-->

-

## Testing

<!--
How was this verified locally? Required at minimum:
- `pytest tests/ -m "not slow"`
- `mypy --strict src/bin_packer_3d/`
- `ruff check . && ruff format --check .`

Add any reproducer commands, dataset attachments, or screenshots that
help a reviewer confirm the fix without rebuilding context.
-->

- [ ] `pytest -m "not slow"` green
- [ ] `mypy --strict src/bin_packer_3d/` green
- [ ] `ruff check` and `ruff format --check` green
- [ ] `pre-commit run --all-files` green

## Breaking changes

<!--
If this PR changes a public surface (Python API, CLI flag, config schema),
describe the migration path. Otherwise write "None" and delete the
checklist below.
-->

- [ ] Migration note added to `CHANGELOG.md`
- [ ] Public API impact documented in `specs/.../contracts/`

## Constitution impact

<!--
Walk the eight principles. Mark NON-NEGOTIABLE gates explicitly:
- I. Contract Honesty (NON-NEG.) — does this preserve docs/runtime alignment?
- II. Test-First Discipline (NON-NEG.) — were tests authored before code?
- III. Automated Quality Gates (NON-NEG.) — do CI checks still pass?
- IV. Reproducibility & Determinism — any new randomness without a seed?
- V. Library Citizenship — any new `print()`, `basicConfig`, import-time side
   effects?
- VI. Documentation as Artefact (NON-NEG.) — public symbols documented?
- VII. Privacy by Default — any new data files vetted?
- VIII. Performance Discipline — any optimisation without a baseline?

Default acceptable answer: "no impact, gates green". Otherwise expand.
-->

- I. Contract Honesty:
- II. Test-First Discipline:
- III. Automated Quality Gates:
- IV. Reproducibility & Determinism:
- V. Library Citizenship:
- VI. Documentation as Artefact:
- VII. Privacy by Default:
- VIII. Performance Discipline:

## Reviewer notes

<!--
Anything else worth flagging — known follow-ups, deferred items, areas
where you want a closer look, etc.
-->
