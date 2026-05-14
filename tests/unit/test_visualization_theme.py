"""Branded Plotly theme regression tests (ADR-006, FR-006 / SC-006).

Five assertions against `bin_packer_3d.visualization.theme`:

1. `apply_theme(fig, "dark")` configures `fig.layout.template` to reflect the
   `BIN_PACKER_3D_DARK` palette and axis-label format.
2. `apply_theme(fig, "light")` similarly applies `BIN_PACKER_3D_LIGHT`.
3. Both `VisualisationStyle` constants advertise the `"{name} (mm)"`
   axis-label format mandated by the contract.
4. `apply_theme` is idempotent — a second call produces the same template
   configuration as the first.
5. `apply_theme` returns the same figure object for fluent chaining.

The theme module is authored in T016; this test is committed in a red
state (T015) beforehand and turns green once T016 lands.
"""

from __future__ import annotations

import plotly.graph_objects as go
from bin_packer_3d.visualization.theme import (
    BIN_PACKER_3D_DARK,
    BIN_PACKER_3D_LIGHT,
    VisualisationStyle,
    apply_theme,
)


def test_apply_theme_dark_uses_dark_palette() -> None:
    """`apply_theme(fig, "dark")` exposes the dark palette via the figure template."""
    fig = go.Figure()
    apply_theme(fig, variant="dark")
    template = fig.layout.template
    assert template is not None, "apply_theme did not set fig.layout.template"
    assert tuple(template.layout.colorway) == BIN_PACKER_3D_DARK.palette, (
        "Dark variant template colorway does not match BIN_PACKER_3D_DARK.palette"
    )


def test_apply_theme_light_uses_light_palette() -> None:
    """`apply_theme(fig, "light")` exposes the light palette via the figure template."""
    fig = go.Figure()
    apply_theme(fig, variant="light")
    template = fig.layout.template
    assert template is not None, "apply_theme did not set fig.layout.template"
    assert tuple(template.layout.colorway) == BIN_PACKER_3D_LIGHT.palette, (
        "Light variant template colorway does not match BIN_PACKER_3D_LIGHT.palette"
    )


def test_axis_label_format_follows_contract() -> None:
    """Both style constants advertise the `"{name} (mm)"` axis-label format."""
    expected = "{name} (mm)"
    assert BIN_PACKER_3D_DARK.axis_label_format == expected
    assert BIN_PACKER_3D_LIGHT.axis_label_format == expected


def test_apply_theme_is_idempotent() -> None:
    """Calling apply_theme twice yields the same template configuration."""
    fig = go.Figure()
    apply_theme(fig, variant="dark")
    first_colorway = tuple(fig.layout.template.layout.colorway)
    apply_theme(fig, variant="dark")
    second_colorway = tuple(fig.layout.template.layout.colorway)
    assert first_colorway == second_colorway, (
        "apply_theme is not idempotent: colorway drifted between calls"
    )


def test_apply_theme_returns_same_figure() -> None:
    """apply_theme mutates in place AND returns the figure for fluent chaining."""
    fig = go.Figure()
    returned = apply_theme(fig, variant="dark")
    assert returned is fig


def test_visualisation_style_is_frozen_dataclass() -> None:
    """VisualisationStyle is immutable per data-model.md (frozen dataclass)."""
    import dataclasses

    assert dataclasses.is_dataclass(VisualisationStyle)
    # Attempting to mutate a frozen dataclass raises FrozenInstanceError.
    try:
        BIN_PACKER_3D_DARK.palette = ("#000000",)  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("BIN_PACKER_3D_DARK is not frozen — mutation succeeded")
