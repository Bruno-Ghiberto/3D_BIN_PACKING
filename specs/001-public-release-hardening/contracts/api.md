# Contract: Public Python API

**Feature**: `001-public-release-hardening`
**Scope**: Names exported from `src/bin_packer_3d/__init__.py`
**Stability**: symbols listed here are covered by the versioning policy in
Constitution §Release workflow. Breaking changes to any symbol listed below
require a MAJOR release (pre-1.0: documented in `CHANGELOG.md`; post-1.0:
MAJOR bump).

## Exported surface

| Symbol | Kind | Module | Since | Status |
|---|---|---|---|---|
| `BinPacker` | class | `bin_packer_3d.cli` → re-exported | 0.1.0 | stabilise |
| `Box` | pydantic model | `models.box` | 0.1.0 | stabilise; `weight` type change 0.2.0 |
| `Bin` | pydantic model | `models.bin` | 0.1.0 | stabilise; `max_weight` Optional 0.2.0 |
| `Placement` | pydantic model | `models.placement` | 0.1.0 | stabilise |
| `PackerConfig` | pydantic settings | `config` | 0.1.0 | stabilise; `constraints`, `seed`, `column_mapping` added 0.3.0 |
| `PackingResult` | pydantic model | `models.result` | 0.2.0 | new |
| `LoadReport` | pydantic model | `models.result` | 0.2.0 | new |
| `ColumnMapping` | dataclass | `data.loaders` | 0.2.0 | new |
| `pack` | function | `bin_packer_3d` | 0.1.0 | stabilise |
| `load_boxes_from_csv` | function | `data.loaders` | 0.1.0 | stabilise; `mapping` param added 0.2.0 |
| `load_boxes_from_excel` | function | `data.loaders` | 0.1.0 | stabilise; `mapping` param added 0.2.0 |
| `plot_packing` | function | `visualization.plotter` | 0.1.0 | stabilise |
| `ALGORITHMS` | dict | `algorithms` | 0.2.0 | new — registry |
| `register` | decorator | `algorithms` | 0.2.0 | new — registry API |
| `get_strategies` | function | `algorithms` | 0.2.0 | new |
| `Constraint` | ABC | `constraints.base` | 0.3.0 | new |
| `ConstraintResult` | dataclass | `constraints.base` | 0.3.0 | new |
| `AllowedOrientations` | class | `constraints.orientation` | 0.3.0 | new |
| `SupportedWeight` | class | `constraints.weight` | 0.3.0 | new |
| `AlgorithmMetadata` | dataclass | `models.metadata` | 0.3.0 | new |
| `BenchmarkResult` | dataclass | `benchmark.results` | 0.3.0 | new |
| `BenchmarkRunner` | class | `benchmark.runner` | 0.3.0 | new |
| `get_logger` | function | `observability` | 0.3.0 | new |

Symbols NOT listed here are **internal** and may change without a version
bump. Consumers importing from submodules not re-exported via
`bin_packer_3d.__init__` do so at their own risk.

## Signatures

### `pack`

```python
def pack(
    boxes: Sequence[Box],
    config: PackerConfig,
) -> PackingResult:
    """Run the configured algorithm against the box list.

    Raises
    ------
    ValueError
        If config.strategy is not a registered algorithm key.
    """
```

### `load_boxes_from_csv`

```python
def load_boxes_from_csv(
    path: str | PathLike[str],
    mapping: ColumnMapping | None = None,
) -> LoadReport:
    """Load boxes from a CSV file.

    Parameters
    ----------
    path : path to the CSV file
    mapping : optional declared column mapping. None means infer using the
        default field names (``length``, ``width``, ``height``, ...).

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    LoaderError
        If required columns (length / width / height) are missing.

    Never raises KeyError on missing optional columns (FR-004).
    """
```

### `load_boxes_from_excel`

Same signature as `load_boxes_from_csv` with `path` being an `.xlsx` file.
Uses `openpyxl` under the hood.

### `plot_packing`

```python
def plot_packing(
    result: PackingResult,
    output_path: str | PathLike[str] | None = None,
    open_browser: bool = False,
) -> Path:
    """Render a 3D visualisation of the packing result.

    Returns
    -------
    Path to the written HTML file.

    Concurrency: output path is derived from ``output_path`` or, when None,
    generated with a timestamp + bin identifier so parallel runs never
    collide (spec §Edge Cases "Concurrent visualiser invocations").
    """
```

### `register`

```python
def register(name: str) -> Callable[[type[BasePacker]], type[BasePacker]]:
    """Register a packer class under the given strategy name.

    Raises
    ------
    ValueError
        If ``name`` is already registered.
    """
```

### `get_logger`

```python
def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger with a NullHandler attached.

    The returned logger is a child of ``bin_packer_3d``; consumers can
    configure it by targeting that parent.
    """
```

## Exception hierarchy

```text
BinPackerError                 # base for library-raised exceptions
 ├─ LoaderError                # raised by data.loaders on fatal input errors
 │   ├─ MissingColumnError     # required column absent
 │   └─ SchemaError            # malformed header / type mismatch at schema level
 ├─ RegistryError              # unknown or duplicate algorithm name
 ├─ ConstraintError            # raised by Constraint subclasses on misconfiguration
 └─ BenchmarkError             # raised by benchmark.runner on instance fetch/parse failures
```

All exceptions carry human-readable messages. Library code never calls
`sys.exit()` (Constitution V).

## Backwards-compatibility table (Phase A)

| Change | Nature | Migration |
|---|---|---|
| `Box.weight: float` → `float \| None` (default `None`) | BREAKING | `Box(weight=0.0)` unchanged; `Box()` now means "unknown" — callers relying on implicit `0.0` must pass it explicitly |
| `PackerConfig.strategy: Literal[...]` → `str` (validated) | non-breaking extension | existing code unaffected; error message improves on unknown values |
| `load_boxes_from_csv` gains `mapping` kwarg | non-breaking | existing calls continue to work with default mapping |
| `PackingResult` dataclass replaces ad-hoc dict return | BREAKING | callers reading `result["placements"]` must switch to `result.placements` |

Every breaking change is recorded in `CHANGELOG.md` under `[0.2.0]` with
the matching migration note.
