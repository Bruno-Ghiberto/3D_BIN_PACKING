"""Branded Plotly theme system for `bin-packer-3d` visualisations (ADR-006).

Exports:

- :class:`VisualisationStyle`: frozen dataclass describing a theme variant.
- :data:`BIN_PACKER_3D_DARK`: dark-variant theme constant.
- :data:`BIN_PACKER_3D_LIGHT`: light-variant theme constant.
- :func:`apply_theme`: mutates a Plotly figure's layout template to a
  named variant; idempotent; returns the figure for fluent chaining.

Both variants share the ColorBrewer Set3 palette (12 entries) which is
colourblind-safe by design (pairwise CIELAB ΔE ≥ 15 under deuteranopia
and protanopia, verified empirically per ADR-012). Only the background,
plot, axis, and overlay colours differ between variants.

Contract: ``specs/002-portfolio-polish/contracts/visualisation-theme.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import plotly.graph_objects as go

# ColorBrewer Set3 (12 entries). Universally documented; the palette
# module (T019) carries the canonical SET3 alongside the hash-indexed
# `colour_for_box`. The tuple here mirrors that constant so theme.py
# remains importable without a dependency on palette.py.
SET3: tuple[str, ...] = (
    "#D9D9D9",  # grey
    "#FFED6F",  # yellow
    "#80B1D3",  # blue
    "#FB8072",  # salmon
    "#BEBADA",  # lavender
    "#FDB462",  # orange
    "#8DD3C7",  # turquoise
    "#B3DE69",  # green
    "#FCCDE5",  # pink
    "#FFFFB3",  # cream
    "#BC80BD",  # purple
    "#CCEBC5",  # mint
)
# Ordering matches `bin_packer_3d.visualization.palette.SET3` and is
# tuned for max min adjacent CIELAB dE under deuteranopia + protanopia
# simulation (~39.80; well above the ADR-012 floor of 15).

_AXIS_LABEL_FORMAT = "{name} (mm)"

_HOVER_TEMPLATE = (
    "<b>%{customdata[0]}</b><br>"
    "Position: x=%{x}, y=%{y}, z=%{z} mm<br>"
    "Size: %{customdata[1]} x %{customdata[2]} x %{customdata[3]} mm"
    "<extra></extra>"
)

_TITLE_FORMAT = "{algorithm} — {dataset}"

_STATS_OVERLAY_LAYOUT: dict[str, Any] = {
    "x": 1.02,
    "y": 1.0,
    "xref": "paper",
    "yref": "paper",
    "showarrow": False,
    "align": "left",
    "font": {"family": "monospace", "size": 11},
    "bgcolor": "rgba(255,255,255,0.75)",
    "bordercolor": "rgba(0,0,0,0.15)",
    "borderwidth": 1,
    "borderpad": 6,
}


@dataclass(frozen=True)
class VisualisationStyle:
    """Branded Plotly theme configuration applied to every emitted figure.

    Attributes:
        name: Identifier; one of ``"bin_packer_3d_dark"`` or
            ``"bin_packer_3d_light"``.
        palette: 12 hex-string colours from ColorBrewer Set3.
            Deterministic, colourblind-safe per FR-037.
        axis_label_format: f-string template for axis titles, e.g.
            ``"{name} (mm)"``.
        hover_template: Plotly hover string with ``%{customdata[*]}``
            fields.
        title_format: f-string template for figure title, e.g.
            ``"{algorithm} — {dataset}"``.
        stats_overlay_layout: Plotly annotation block defining the
            stats panel layout (algorithm, boxes placed, utilisation,
            runtime).
    """

    name: Literal["bin_packer_3d_dark", "bin_packer_3d_light"]
    palette: tuple[str, ...]
    axis_label_format: str
    hover_template: str
    title_format: str
    stats_overlay_layout: dict[str, Any]


BIN_PACKER_3D_DARK = VisualisationStyle(
    name="bin_packer_3d_dark",
    palette=SET3,
    axis_label_format=_AXIS_LABEL_FORMAT,
    hover_template=_HOVER_TEMPLATE,
    title_format=_TITLE_FORMAT,
    stats_overlay_layout=_STATS_OVERLAY_LAYOUT,
)

BIN_PACKER_3D_LIGHT = VisualisationStyle(
    name="bin_packer_3d_light",
    palette=SET3,
    axis_label_format=_AXIS_LABEL_FORMAT,
    hover_template=_HOVER_TEMPLATE,
    title_format=_TITLE_FORMAT,
    stats_overlay_layout=_STATS_OVERLAY_LAYOUT,
)


def _build_template(style: VisualisationStyle) -> go.layout.Template:
    """Materialise a Plotly Template from a VisualisationStyle.

    Background / surface colours differ per variant; the palette and
    typographic conventions are shared.
    """
    is_dark = style.name == "bin_packer_3d_dark"
    paper = "#1e1e1e" if is_dark else "#ffffff"
    plot = "#2a2a2a" if is_dark else "#f5f5f5"
    font_colour = "#e6e6e6" if is_dark else "#1f1f1f"
    grid_colour = "rgba(255,255,255,0.12)" if is_dark else "rgba(0,0,0,0.12)"

    axis_defaults = {
        "showgrid": True,
        "gridcolor": grid_colour,
        "zerolinecolor": grid_colour,
        "title": {"font": {"color": font_colour}},
    }

    return go.layout.Template(
        layout={
            "colorway": list(style.palette),
            "paper_bgcolor": paper,
            "plot_bgcolor": plot,
            "font": {"color": font_colour, "family": "sans-serif"},
            "scene": {
                "xaxis": axis_defaults,
                "yaxis": axis_defaults,
                "zaxis": axis_defaults,
                "bgcolor": plot,
            },
            "xaxis": axis_defaults,
            "yaxis": axis_defaults,
        }
    )


_DARK_TEMPLATE = _build_template(BIN_PACKER_3D_DARK)
_LIGHT_TEMPLATE = _build_template(BIN_PACKER_3D_LIGHT)


def apply_theme(fig: go.Figure, variant: Literal["dark", "light"] = "dark") -> go.Figure:
    """Apply the bin-packer-3d branded theme to a Plotly figure.

    Mutates ``fig.layout.template`` to the named variant and returns the
    same figure object for fluent chaining. Idempotent — calling twice
    with the same variant produces the same configuration.

    Args:
        fig: The Plotly figure to brand.
        variant: ``"dark"`` (default) or ``"light"``.

    Returns:
        The same ``fig`` instance, with its layout template updated.

    Raises:
        ValueError: if ``variant`` is not one of the supported names.
    """
    if variant == "dark":
        fig.update_layout(template=_DARK_TEMPLATE)
    elif variant == "light":
        fig.update_layout(template=_LIGHT_TEMPLATE)
    else:
        raise ValueError(f'variant must be "dark" or "light", got {variant!r}')
    return fig
