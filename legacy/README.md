# legacy/

Preserved for historical reference only. **Not** part of the
supported `bin-packer-3d` package.

## What's in here

Four Python scripts from the original 3D-BPP research notebook
that preceded the current library:

- `MAIN.py` — entry point that wired the workflow end-to-end.
- `Utils.py` — data loading + preprocessing helpers for the CNH
  packaging datasets.
- `AdvancedHeuristicPacker.py` — prototype of the shelf + FFD
  hybrid that evolved into `src/bin_packer_3d/algorithms/`.
- `Plotter.py` — prototype visualiser that evolved into
  `src/bin_packer_3d/visualization/plotter.py`.

## Why these files are NOT supported

- Paths are hardcoded for the original author's Windows
  environment (`C:\Users\bghiberto\source\repos\...`). They will
  not run on another machine without edits.
- No tests, no docstrings to the project standard, no type
  annotations to the project standard.
- API design superseded by `src/bin_packer_3d/`. Anything worth
  keeping has been re-implemented or will be (e.g. the
  per-product weight table lookups will inform US7's constraint
  framework).
- Excluded from built wheel / sdist via `pyproject.toml`
  `[tool.hatch.build] exclude = ["legacy/", ...]`.

## Why keep them at all

Historical reference during spec-01 implementation — several
design decisions in `src/bin_packer_3d/` (FFD preprocessing
order, shelf ceiling semantics, column-mapping fallbacks for
Spanish CSV headers) trace back to choices made in these scripts.
Keeping the original code one `git log --follow` away is cheaper
than rediscovering why each choice was made.

Browse them as archaeology, not as runnable code.

## Provenance

The scripts lived under `CODE/` at the repository root through
the project's Alpha (v0.1.0) lifecycle. Invocation 4 of spec-01
(`001-public-release-hardening`, T081) relocated them to this
`legacy/` directory. The pre-move state is preserved as the
annotated git tag **`legacy-code-preserved`** — `git checkout
legacy-code-preserved` recovers the exact pre-move layout if
ever needed.

---

See `src/bin_packer_3d/` for the current implementation and
`docs/` for its documentation.
