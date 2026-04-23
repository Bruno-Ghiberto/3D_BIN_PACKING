# Quickstart: `bin-packer-3d`

**Feature**: `001-public-release-hardening`
**Audience**: first-time user on a clean machine with Python 3.11+
**Spec references**: US3 AC4 (five-command setup), FR-027, SC-007 (git clone
to passing tests in five commands or fewer).

This quickstart is the ground truth for the "5-command flow" acceptance
criterion. Each command must:

- exit `0` on a clean install (no missing deps, no import errors),
- produce human-readable output or a browser visualisation within 5 s on a
  commodity laptop (no GPU required),
- print a `--help` that names the purpose, required flags, and one example.

The commands are verified in CI via `tests/integration/test_cli.py`.

---

## Scenario A — Library user (via PyPI)

The end-user install path. Five commands. Zero boilerplate.

```bash
# 1. Install from PyPI (OIDC-published wheel)
pip install bin-packer-3d

# 2. Verify the install
bin-packer info

# 3. Initialise a sample config in CWD
bin-packer init --output packing_config.json

# 4. Download a sample CSV (ships with the package; copy from examples)
python -c "from importlib.resources import files; \
  print(files('bin_packer_3d').joinpath('examples/sample_boxes.csv').read_text())" \
  > sample_boxes.csv

# 5. Pack the sample dataset and render the 3D visualisation
bin-packer pack --input sample_boxes.csv --config packing_config.json --visualise
```

**Expected output after step 5** (shape — exact numbers depend on input):

```text
Packing with 'ffd' against 40 boxes in bin (100.0, 80.0, 60.0)
  Placed: 38 / 40 (95.0%)
  Bins used: 2
  Volume utilisation: 81.2%
  Runtime: 0.042 s
  Visualisation: ./output/packing_bin1_20260423T143000Z.html
```

The HTML file opens in the user's default browser (Plotly interactive
plot).

---

## Scenario B — Contributor (from git clone)

The `CONTRIBUTING.md` setup path. Five commands from `git clone` to a
passing test run (FR-027, SC-007).

```bash
# 1. Clone and enter
git clone https://github.com/Bruno-Ghiberto/3D_BIN_PACKING.git
cd 3D_BIN_PACKING

# 2. Install dev extras in editable mode (creates a local venv first if desired)
pip install -e '.[dev,docs]'

# 3. Install pre-commit hooks (mirrors CI — lint, format, type check)
pre-commit install

# 4. Run the test suite
pytest

# 5. Build the docs locally (live reload at http://127.0.0.1:8000)
mkdocs serve
```

Step 4 must exit `0` with `pytest-cov` reporting ≥ 90 % line coverage
(Constitution III, Spec FR-012).

---

## Python API scenario (library embedding)

Equivalent to Scenario A but from Python code — for consumers embedding the
library inside a larger application.

```python
from bin_packer_3d import (
    pack, load_boxes_from_csv, PackerConfig, plot_packing,
)
from bin_packer_3d.constraints import AllowedOrientations  # 0.3.0+

# 1. Load boxes
report = load_boxes_from_csv("sample_boxes.csv")
if report.rejected_rows:
    print(f"Warning: {len(report.rejected_rows)} rows rejected")

# 2. Configure a run — with a future stability constraint added, the list
#    grows without breaking algorithm code (FR-063).
config = PackerConfig(
    strategy="extreme_point",              # 0.3.0+ — requires registry
    bin_dimensions=(100.0, 80.0, 60.0),
    seed=42,                               # reproducibility (FR-043)
    constraints=[
        AllowedOrientations(),             # reads per-box preference
    ],
)

# 3. Pack
result = pack(report.boxes, config)

print(f"Placed {len(result.placements)} / {len(report.boxes)}")
print(f"Bins used: {result.bins_used}")
print(f"Volume utilisation: {result.volume_utilisation:.1%}")

# 4. Visualise (optional)
html_path = plot_packing(result, output_path="./my_pack.html")
print(f"Visualisation: {html_path}")
```

---

## Docker scenario

For users who do not want Python on their host (FR-072).

```bash
# Pull and inspect
docker pull ghcr.io/bruno-ghiberto/bin-packer-3d:latest
docker run --rm ghcr.io/bruno-ghiberto/bin-packer-3d:latest info

# Pack against a mounted CSV
docker run --rm \
  -v "$PWD:/work" -w /work \
  ghcr.io/bruno-ghiberto/bin-packer-3d:latest \
  pack --input sample_boxes.csv --config packing_config.json
```

The image uses a multi-stage build, runs as a non-root user, and has no
network dependency at runtime.

---

## Validation checklist

When a new release is cut, Scenarios A and B are executed on a clean Linux
runner as part of the release workflow's smoke test. Scenarios that fail
the 5-second response target on the reference laptop (16 GB RAM,
mid-range CPU) block the release.

- [ ] `pip install bin-packer-3d` succeeds on Python 3.11, 3.12, 3.13, 3.14.
- [ ] `bin-packer info` prints ≥ 5 registered algorithms (post-v0.3.0).
- [ ] `bin-packer pack --visualise` produces an HTML file and exits 0.
- [ ] `git clone && pip install -e .[dev,docs] && pre-commit install &&
      pytest && mkdocs serve` completes in ≤ 5 commands.
- [ ] `pytest` reports ≥ 90 % coverage.
- [ ] `docker run <image> info` prints algorithms and exits 0.
