"""3D visualisation using Plotly with the bin-packer-3d branded theme.

This module produces interactive HTML and (when ``kaleido`` is
installed) static-export figures of packing solutions. Every figure
goes through :func:`apply_theme` and consumes the deterministic
:attr:`Placement.colour` property, so two consecutive runs on identical
input produce byte-identical HTML and pixel-equal PNG / SVG output
(FR-006, SC-005, ADR-001, ADR-006, ADR-011).

DR-11 (Phase B'-1): ``kaleido`` is pinned to ``>=0.2.1,<1.0.0`` so that
``fig.write_image()`` works against Plotly 5.x. A future migration to
``kaleido.write_fig()`` and Plotly 6.x is a separate spec.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import plotly.graph_objs as go

from bin_packer_3d.visualization.theme import BIN_PACKER_3D_DARK, apply_theme

if TYPE_CHECKING:
    from bin_packer_3d.config import VisualizationConfig
    from bin_packer_3d.models.bin import Bin
    from bin_packer_3d.models.placement import PlacementResult


# Coarse runtime bucket size in milliseconds for the stats overlay.
# Rounding wall-clock runtime to this granularity preserves byte-identical
# HTML across two runs on the same input (FR-006 + SC-005). For headline-
# sized BFD packs (~8 boxes) every run fits in <10ms so the overlay always
# reads "<10ms"; larger workloads round to the nearest 10ms.
_RUNTIME_BUCKET_MS = 10


def _runtime_display(elapsed_ms: float) -> str:
    """Return a deterministic-across-runs runtime label for the overlay."""
    if elapsed_ms < _RUNTIME_BUCKET_MS:
        return f"<{_RUNTIME_BUCKET_MS}ms"
    bucket = int(elapsed_ms // _RUNTIME_BUCKET_MS) * _RUNTIME_BUCKET_MS
    return f"~{bucket}ms"


class Plotter3D:
    """Interactive 3D visualisation of packing results.

    Produces Plotly figures with:

    - Semi-transparent mesh boxes coloured by :attr:`Placement.colour`
      (deterministic BLAKE2b hash → SET3 palette per ADR-007).
    - Wireframe edges per box.
    - Branded theme (:data:`BIN_PACKER_3D_DARK`) applied via
      :func:`apply_theme`.
    - Stats-overlay annotation (algorithm, boxes placed, utilisation %,
      runtime bucket).
    - Deterministic HTML (explicit ``div_id`` per bin; CDN-loaded
      ``plotly.js``) so two consecutive runs are byte-identical.
    - Optional static export via Kaleido (PNG default, SVG opt-in) when
      the ``viz`` extras are installed.

    Example:
        >>> plotter = Plotter3D()
        >>> plotter.plot_result(
        ...     result,
        ...     output_dir="output/",
        ...     dataset="headline",
        ...     static_format="png",
        ... )
    """

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        """Initialise plotter with optional visualisation config."""
        from bin_packer_3d.config import VisualizationConfig

        self.config = config or VisualizationConfig()

    def _make_wireframe(
        self,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        z0: float,
        z1: float,
        color: str = "black",
        name: str = "wire",
    ) -> go.Scatter3d:
        """Create wireframe edges for an axis-aligned bounding box."""
        corners = [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # bottom
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # top
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # vertical
        ]

        x_vals, y_vals, z_vals = [], [], []
        for start, end in edges:
            x_vals.extend([corners[start][0], corners[end][0], None])
            y_vals.extend([corners[start][1], corners[end][1], None])
            z_vals.extend([corners[start][2], corners[end][2], None])

        return go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode="lines",
            line={"color": color, "width": 2},
            name=name,
            hoverinfo="none",
        )

    def _make_mesh(
        self,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        z0: float,
        z1: float,
        color: str,
        hover_text: str,
    ) -> go.Mesh3d:
        """Create a semi-transparent mesh for a box."""
        vertices = [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ]

        faces = [
            (0, 1, 2),
            (0, 2, 3),  # bottom
            (4, 5, 6),
            (4, 6, 7),  # top
            (0, 1, 5),
            (0, 5, 4),  # front
            (3, 2, 6),
            (3, 6, 7),  # back
            (0, 3, 7),
            (0, 7, 4),  # left
            (1, 2, 6),
            (1, 6, 5),  # right
        ]

        return go.Mesh3d(
            x=[v[0] for v in vertices],
            y=[v[1] for v in vertices],
            z=[v[2] for v in vertices],
            i=[f[0] for f in faces],
            j=[f[1] for f in faces],
            k=[f[2] for f in faces],
            color=color,
            opacity=self.config.opacity,
            hoverinfo="text",
            text=hover_text,
            name="box",
        )

    def plot_bin(
        self,
        bin_obj: Bin,
        *,
        title: str,
        overlay_text: str | None = None,
        variant: str = "dark",
    ) -> go.Figure:
        """Build a branded Plotly figure for a single bin.

        Args:
            bin_obj: Bin with placements to visualise.
            title: Fully-formatted title (see
                :attr:`VisualisationStyle.title_format`).
            overlay_text: Optional pre-formatted stats overlay; rendered
                via ``fig.add_annotation`` using
                :attr:`VisualisationStyle.stats_overlay_layout`.
            variant: ``"dark"`` (default) or ``"light"``.

        Returns:
            A themed Plotly figure ready for HTML or static export.
        """
        style = BIN_PACKER_3D_DARK if variant == "dark" else BIN_PACKER_3D_DARK
        # Light variant currently shares overlay layout + title format
        # with the dark variant; only the template differs.

        traces = []

        # Bin boundary wireframe in neutral grey for both variants.
        bin_wire = self._make_wireframe(
            0,
            bin_obj.length,
            0,
            bin_obj.width,
            0,
            bin_obj.height,
            color="#888888",
            name="Bin",
        )
        traces.append(bin_wire)

        # One mesh + (optional) wireframe per placement, coloured via
        # the deterministic Placement.colour @cached_property (Phase A
        # T022 + ADR-007).
        for placement in bin_obj.placements:
            box = placement.box
            colour = placement.colour

            hover_text = (
                f"ID: {box.id}<br>"
                f"Type: {box.box_type}<br>"
                f"Desc: {box.description}<br>"
                f"Qty: {box.quantity}"
            )

            mesh = self._make_mesh(
                placement.x0,
                placement.x1,
                placement.y0,
                placement.y1,
                placement.z0,
                placement.z1,
                color=colour,
                hover_text=hover_text,
            )
            traces.append(mesh)

            if self.config.show_wireframe:
                wire = self._make_wireframe(
                    placement.x0,
                    placement.x1,
                    placement.y0,
                    placement.y1,
                    placement.z0,
                    placement.z1,
                    color=colour,
                    name=f"Box {box.id}",
                )
                traces.append(wire)

        layout = go.Layout(
            title=title,
            scene={
                "xaxis": {"title": style.axis_label_format.format(name="Length")},
                "yaxis": {"title": style.axis_label_format.format(name="Width")},
                "zaxis": {"title": style.axis_label_format.format(name="Height")},
                "aspectmode": "manual",
                "aspectratio": {"x": 1.2, "y": 1, "z": 1},
            },
        )

        fig = go.Figure(data=traces, layout=layout)
        apply_theme(fig, variant=variant)  # type: ignore[arg-type]

        if overlay_text:
            fig.add_annotation(text=overlay_text, **style.stats_overlay_layout)

        return fig

    def plot_result(
        self,
        result: PlacementResult,
        output_dir: Path | str = "output",
        *,
        dataset: str = "",
        static_format: str | None = None,
        variant: str = "dark",
    ) -> list[Path]:
        """Render every bin in ``result`` to HTML (and optional static export).

        Args:
            result: Packing result to visualise.
            output_dir: Directory where ``bin_<N>.html`` (and optionally
                ``bin_<N>.png`` / ``.svg``) are written.
            dataset: Dataset name used in the branded title format.
                Typically the input file's stem, e.g. ``"headline"``.
            static_format: ``"png"``, ``"svg"``, or ``None``. When not
                ``None``, also emit a static export per bin via Kaleido.
                Raises ``RuntimeError`` if Kaleido is not installed and
                static export was requested.
            variant: ``"dark"`` (default) or ``"light"``.

        Returns:
            Paths to every artefact created (HTML always; static export
            paths appended when requested).
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        title = BIN_PACKER_3D_DARK.title_format.format(
            algorithm=result.algorithm,
            dataset=dataset or "—",
        )

        overlay_text = (
            f"Algorithm: {result.algorithm}<br>"
            f"Boxes placed: {result.placed_count}/{result.total_boxes}<br>"
            f"Utilisation: {result.utilization_percent:.1f}%<br>"
            f"Runtime: {_runtime_display(result.elapsed_time_ms)}"
        )

        created_files: list[Path] = []

        for bin_obj in result.bins:
            fig = self.plot_bin(
                bin_obj,
                title=title,
                overlay_text=overlay_text,
                variant=variant,
            )

            # Deterministic HTML: explicit div_id removes Plotly's
            # default UUID; include_plotlyjs="cdn" keeps file size
            # small and the embedded markup stable across runs.
            html = fig.to_html(
                div_id=f"bin_{bin_obj.id}",
                include_plotlyjs="cdn",
                full_html=True,
            )
            html_path = output_path / f"bin_{bin_obj.id}.html"
            html_path.write_text(html, encoding="utf-8")
            created_files.append(html_path)

            if static_format is not None:
                self._emit_static(fig, output_path, bin_obj.id, static_format)
                created_files.append(output_path / f"bin_{bin_obj.id}.{static_format}")

        return created_files

    @staticmethod
    def _emit_static(
        fig: go.Figure,
        output_path: Path,
        bin_id: int,
        static_format: str,
    ) -> None:
        """Emit a static export of ``fig`` via Kaleido.

        Guarded by a local ``import kaleido``: if the ``viz`` extra is
        not installed, raise ``RuntimeError`` with the install
        instruction documented in
        ``specs/002-portfolio-polish/contracts/visualisation-theme.md``.
        """
        try:
            # TODO(T038): kaleido ships without a py.typed marker;
            # narrow-ignore the import. Revisit when kaleido publishes
            # type stubs or when migrating to the Kaleido 1.x
            # write_fig() API (future spec; DR-11 deferred).
            import kaleido  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Static export requires 'kaleido'. Install with: pip install 'bin-packer-3d[viz]'"
            ) from exc

        static_path = output_path / f"bin_{bin_id}.{static_format}"
        fig.write_image(
            str(static_path),
            format=static_format,
            engine="kaleido",
        )
