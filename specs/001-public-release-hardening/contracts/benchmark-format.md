# Contract: Benchmark Result JSON Schema

**Feature**: `001-public-release-hardening`
**Model**: `bin_packer_3d.benchmark.results.BenchmarkResult`
**Serialiser**: `dataclasses.asdict(result)` → `json.dumps(..., indent=2,
default=str)`.

This document is the versioned schema for benchmark result JSON files
attached to CI builds and published at `docs/benchmarks/results/`.
Downstream consumers (the docs site regenerator, the release notes
template, external reviewers) depend on this shape.

## Top-level document

A benchmark run produces one of two top-level shapes:

### `SingleResult` — one algorithm, one instance

```json
{
  "schema_version": "1",
  "kind": "single",
  "result": { /* BenchmarkResult */ }
}
```

### `ComparisonTable` — multiple algorithms × one or more instances

```json
{
  "schema_version": "1",
  "kind": "comparison",
  "results": [ /* BenchmarkResult, ... */ ],
  "summary": {
    "n_algorithms": 5,
    "n_instances": 1,
    "seed": 0,
    "generated_at": "2026-04-23T14:30:00Z",
    "bin_packer_version": "0.3.0"
  }
}
```

## `BenchmarkResult` object

```json
{
  "algorithm": "extreme_point",
  "instance": "BR1",
  "n_boxes": 120,
  "n_bins_used": 3,
  "volume_utilisation": 0.837,
  "success_rate": 100.0,
  "elapsed_seconds": 0.128,
  "metadata": {
    "name": "extreme_point",
    "version": "0.3.0",
    "parameters": {
      "bin_dimensions": [587, 233, 220],
      "allow_rotation": true
    },
    "seed": 0,
    "timestamp": "2026-04-23T14:30:00.123456Z"
  },
  "notes": ""
}
```

### Field reference

| Field | Type | Required | Notes |
|---|---|---|---|
| `algorithm` | string | YES | registry key |
| `instance` | string | YES | e.g. `"BR1"` — matches `BenchmarkInstance.name` |
| `n_boxes` | int | YES | input box count |
| `n_bins_used` | int | YES | `>= 0` |
| `volume_utilisation` | number | YES | `[0.0, 1.0]` |
| `success_rate` | number | YES | `[0.0, 100.0]`, percent of boxes placed |
| `elapsed_seconds` | number | YES | `>= 0` |
| `metadata.name` | string | YES | same as `algorithm` |
| `metadata.version` | string | YES | `bin_packer_3d.__version__` |
| `metadata.parameters` | object | YES | captured from `PackerConfig` |
| `metadata.seed` | int \| null | YES | null only when `seed is None` was resolved to a random value — then `notes` carries the chosen seed |
| `metadata.timestamp` | string (ISO 8601 UTC) | YES | RFC 3339 compliant |
| `notes` | string | NO | default `""` |

## Schema versioning

Current: `"schema_version": "1"`.

Breaking changes to this schema follow the same policy as the config
schema: field renames, removals, and type changes bump the schema version
to `"2"`. Consumers MUST verify `schema_version` before parsing.

## CI artefact naming

- Per-push BR1 benchmark: `benchmark-br1-${{ github.sha }}.json` —
  attached to the workflow run.
- Release full suite: `benchmark-br1-br8-${{ github.ref_name }}.json` —
  attached to the GitHub Release and the PyPI publish log.

## Reproducibility contract

Given:
- identical `bin_packer_version`,
- identical `instance` contents,
- identical `seed`,
- identical `metadata.parameters`,

the resulting `BenchmarkResult` MUST be bit-identical (FR-043).
`volume_utilisation`, `elapsed_seconds`, and `metadata.timestamp` are
NOT compared when asserting reproducibility — only the placement decisions
(`n_bins_used`, `n_boxes`, and internal placement list serialised
alongside when present) must match exactly.
