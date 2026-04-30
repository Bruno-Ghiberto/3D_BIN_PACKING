# Contract: CLI

**Feature**: `001-public-release-hardening`
**Entry point**: `bin-packer` (installed via `[project.scripts]` in
`pyproject.toml`). Implemented in `src/bin_packer_3d/cli.py` using `click`.
**Module invocation**: `python -m bin_packer_3d` is equivalent to
`bin-packer`.

## Global options

| Flag | Type | Default | Since | Notes |
|---|---|---|---|---|
| `--verbose` / `-v` | flag | off | 0.3.0 | sets effective log level to DEBUG (FR-051) |
| `--quiet` / `-q` | flag | off | 0.3.0 | sets effective log level to WARNING |
| `--version` | flag | — | 0.1.0 | prints `bin_packer_3d.__version__` and exits |
| `--help` | flag | — | 0.1.0 | prints usage |

`--verbose` and `--quiet` are mutually exclusive — invoking both exits
non-zero with a clear message.

## Subcommands

### `bin-packer info`

Print the registered algorithms with complexity class and short description.

**Flags**:

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--format` | choice(`text`, `json`) | `text` | 0.3.0 |

**Output (text)**:

```text
Registered algorithms (5):
  ffd                 First-Fit Decreasing (volume)           O(n log n)
  shelf               Shelf-based                             O(n log n)
  bfd                 Best-Fit Decreasing                     O(n log n)
  extreme_point       Extreme Point (Crainic 2008)            O(n^2)
  maximal_rectangles  Maximal Rectangles                      O(n^3)
```

The listing is generated from `ALGORITHMS` at runtime — never hardcoded
(FR-003).

**Exit codes**: `0` success; `1` if the registry is somehow empty (import
failure).

---

### `bin-packer init`

Write a sample `PackerConfig` JSON file to disk.

**Flags**:

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--output` / `-o` | path | `packing_config.json` | written relative to CWD |

**Behaviour**: writes a pydantic-serialised `PackerConfig` with all
defaults. Never overwrites an existing file without `--force`.

**Exit codes**: `0` success; `1` destination exists and `--force` not
passed; `2` permission denied.

---

### `bin-packer pack`

Run packing against an input file.

**Flags**:

| Flag | Type | Default | Required | Notes |
|---|---|---|---|---|
| `--input` / `-i` | path | — | YES | CSV or XLSX |
| `--config` / `-c` | path | `packing_config.json` | NO | loaded if exists, defaults otherwise |
| `--algorithm` / `-a` | str | (config value) | NO | overrides `config.strategy`; must be registered |
| `--visualise` / `--no-visualise` | flag | `--no-visualise` | NO | produces HTML plot when set |
| `--output-dir` / `-o` | path | `./output/` | NO | where visualisations and reports are written |
| `--explain` | flag | off | NO | emits per-box placement trace at DEBUG (FR-054) |
| `--seed` | int | (config or random) | NO | forwarded to `PackerConfig.seed` (FR-043) |

**Output**: packing summary to stdout (Rich-formatted when TTY); when
`--visualise` is set, an HTML file is written to `--output-dir` with a
name including the timestamp + bin identifier to avoid collisions.

**Exit codes**:

| Code | Meaning |
|---|---|
| 0 | pack completed (irrespective of `success_rate`) |
| 1 | input not found / invalid |
| 2 | config not found / invalid (e.g. unknown algorithm) |
| 3 | internal error (unhandled exception — stack trace printed) |

---

### `bin-packer benchmark`

Run one or more registered algorithms against one or more reference
instances and emit a comparison.

**Flags** (all since 0.3.0):

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--instance` / `-i` | choice(`BR1`..`BR8`, `all`) | `BR1` | reference instance; `all` = BR1..BR8 |
| `--algorithm` / `-a` | str (repeatable) | all registered | filter to a subset |
| `--seed` | int | `0` | bit-identical output with same seed (FR-043) |
| `--format` | choice(`text`, `json`, `markdown`) | `text` | FR-042 |
| `--output` / `-o` | path | stdout | when set, writes to the path instead |

**Output (text)**:

```text
Instance: BR1 (seed=0)

Algorithm            Util %   Bins  Success %  Runtime (s)
ffd                   78.4      3      100.0       0.042
shelf                 75.1      3      100.0       0.031
bfd                   80.2      3      100.0       0.045
extreme_point         83.7      3      100.0       0.128
maximal_rectangles    85.1      3      100.0       0.203
```

**Exit codes**: `0` success; `1` unknown instance or algorithm name
(error message lists every registered name, FR-047); `2` instance fetch
failed (when license-gated and downloader unreachable).

---

## Conventional-Commits PR title enforcement

Orthogonal to CLI contract, but part of the release surface: PR titles on
this repo MUST match Conventional Commits (`feat`, `fix`, `chore`, `docs`,
`refactor`, `test`, `ci`, `build`, `perf`, `style`, `revert`). Enforced by
`.github/workflows/pr-title.yml`.
