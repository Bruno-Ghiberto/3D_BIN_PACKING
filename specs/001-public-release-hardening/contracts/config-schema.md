# Contract: `PackerConfig` Schema

**Feature**: `001-public-release-hardening`
**Model**: `bin_packer_3d.config.PackerConfig` (pydantic v2 settings model)

This document is the versioned reference for the configuration schema.
Any field rename, type change, or default removal is a breaking change
requiring a MAJOR release (or, pre-1.0, a `CHANGELOG.md` entry under
`[x.y.0]` with a migration note).

## Field reference

| Field | Type | Default | Since | Validator |
|---|---|---|---|---|
| `strategy` | `str` | `"ffd"` | 0.1.0 (behaviour change 0.2.0) | must be a key of `ALGORITHMS` |
| `bin_dimensions` | `tuple[float, float, float]` | required | 0.1.0 | each element `> 0` |
| `allow_rotation` | `bool` | `True` | 0.1.0 | deprecated 0.3.0 — prefer per-box `allowed_orientations` |
| `constraints` | `list[Constraint]` | `[]` | 0.3.0 | each must be a `Constraint` instance |
| `column_mapping` | `ColumnMapping \| None` | `None` | 0.2.0 | `None` = infer defaults |
| `seed` | `int \| None` | `None` | 0.3.0 | `None` = random per run (printed on use) |

## Example (`packing_config.json` from `bin-packer init`)

```json
{
  "strategy": "ffd",
  "bin_dimensions": [100.0, 80.0, 60.0],
  "allow_rotation": true,
  "constraints": [],
  "column_mapping": null,
  "seed": null
}
```

## Validation errors

| Error | Cause | Message shape |
|---|---|---|
| `strategy` unknown | value not in `ALGORITHMS` | `unknown strategy '{v}'. Registered: [...]` |
| `bin_dimensions` non-positive | any element `<= 0` | `bin dimensions must be positive; got {v}` |
| `constraints` item type | element is not a `Constraint` instance | pydantic default message |

Validation occurs at config-construction time. `pack()` does not re-validate
the config; callers may serialise/deserialise through pydantic without
losing validator guarantees.

## Breaking-change policy

- **Field added** with default value → MINOR (non-breaking).
- **Field removed** → MAJOR.
- **Field type changed** (e.g. `float` → `int`) → MAJOR.
- **Field default changed** in a way that alters behaviour of existing
  configs → MAJOR (rationale: prior configs silently change semantics).
- **Validator added** that rejects previously-accepted values → MAJOR.
- **Validator relaxed** (accepts more) → MINOR.

## Cross-references

- Field semantics: [data-model.md §`PackerConfig`](../data-model.md).
- CLI defaults: [cli.md](./cli.md).
- Algorithm keys registered: [api.md §`ALGORITHMS`](./api.md).
