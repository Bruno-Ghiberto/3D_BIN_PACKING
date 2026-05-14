# Contract: Headline Dataset Generator

**Owner**: spec-02 (Portfolio Polish) | **Status**: STABLE at `v0.3.0-rc1` | **Source ADRs**: ADR-009 | **Spec ref**: FR-034

## Purpose

Specify the deterministic generator that produces the procedurally generated headline dataset used by all README screenshots, the demo command, and the visualisation gallery.

## Invocation

```bash
python scripts/generate_headline_dataset.py \
  --seed examples/headline.seed \
  --out examples/headline.csv
```

## Inputs

### Seed file (`examples/headline.seed`)

JSON document, committed to the repository.

```json
{
  "seed": 42,
  "target_utilisation": 0.65,
  "bin_dimensions": [860.0, 890.0, 1040.0],
  "n_boxes": 50,
  "min_box_volume_mm3": 50000.0
}
```

Schema validation (enforced by the generator at read time):

- `seed`: non-negative integer.
- `target_utilisation`: float in (0, 1).
- `bin_dimensions`: list of exactly 3 positive floats (length, width, height in mm).
- `n_boxes`: positive integer.
- `min_box_volume_mm3`: positive float strictly less than the product of `bin_dimensions`.

## Outputs

### Generated CSV (`examples/headline.csv`)

Format matches the existing project loader convention (`DATASETS/sample_boxes.csv`):

```csv
ITEM,W,H,L,CANTIDAD,CAJA,DESCRIPCION
HEAD-001,420.0,310.0,260.0,1,TYPE_A,Generated box 001
HEAD-002,180.0,140.0,220.0,1,TYPE_B,Generated box 002
...
```

Columns (header row required):

- `ITEM` (string): unique identifier; format `HEAD-NNN` zero-padded to 3 digits.
- `W` (float): width in mm.
- `H` (float): height in mm.
- `L` (float): length in mm.
- `CANTIDAD` (int): always `1` (one row per box; FR-034 demands per-box rows, not aggregated).
- `CAJA` (string): box type label cycled from a documented set (e.g. `TYPE_A`, `TYPE_B`, `TYPE_C`).
- `DESCRIPCION` (string): `Generated box <ITEM>` for human readability.

## Behavioural Guarantees

1. **Byte-identical regeneration (Principle IV)**: Same seed file → byte-identical CSV bytes. Verified by `tests/unit/test_dataset_generator.py`.
2. **Acceptance (FR-034)**: The generated dataset packed by BFD on default bin dimensions (860×890×1040 mm) yields overall utilisation ≥ 60%.
3. **Cross-Python determinism (Principle IV)**: The generator uses `random.Random(seed)` and only stdlib operations; the same seed produces the same output on Python 3.11–3.14.
4. **Pure-Python (Principle V)**: No native deps; uses only stdlib + numpy (already a runtime dep).
5. **Runtime budget**: Generator completes in ≤ 5 seconds on commodity laptop.

## Algorithm (ADR-009)

Bin-feasibility-driven recursive guillotine cuts:

1. Compute target packed volume `V_pack = target_utilisation × product(bin_dimensions)`.
2. Initialise a virtual root region of dimensions `bin_dimensions` and volume `V_pack`'s effective sub-bin (the rest is "air" that never becomes a box).
3. Recursively partition: at each step, pick a random axis (length / width / height), pick a random cut-position within bounds, split the current region into two sub-regions. Repeat until either:
   - `n_boxes` leaf regions exist, OR
   - All remaining regions have volume < `min_box_volume_mm3`.
4. Collect leaf-region dimensions as the box list.
5. Shuffle the box list via seeded `Random.shuffle()` so emission order is non-trivial.
6. Assign IDs (`HEAD-001`, `HEAD-002`, ...) in shuffled order.
7. Write to CSV.

## Failure Modes

| Scenario | Behaviour | Exit code |
|---|---|---|
| Seed file path does not exist | Click `BadParameter` | 2 |
| Seed file is invalid JSON | `json.JSONDecodeError` propagates | 1 |
| Seed-file schema validation fails | `ValueError` with field name | 1 |
| Generated dataset achieves < 60% BFD utilisation (regression) | Generator script exits non-zero; CI test fails | 1 |
| Output path is not writable | `PermissionError` propagates | 1 |

## Test Coverage

- `tests/unit/test_dataset_generator.py`:
  - Run generator twice with the same seed; assert CSV byte-equal.
  - Run BFD on the generated dataset; assert overall utilisation ≥ 0.60.
  - Assert generator completes in < 5 seconds.

## Breaking-Change Policy

Changes to the following are breaking and require a major version bump:

- Renaming the script path or seed file path.
- Changing the seed-file JSON schema (field names, types, semantics).
- Changing the CSV column order, names, or formats.
- Changing the deterministic algorithm in a way that produces different output for the same seed.

Tuning hyperparameters (default `target_utilisation`, default `n_boxes`) within the contract bounds is allowed as long as the seed-file remains the source of truth.

Adding new optional seed-file fields with backward-compatible defaults is a non-breaking minor change.
