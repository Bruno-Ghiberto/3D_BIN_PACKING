"""3D visualization using Plotly.

This module provides interactive 3D visualizations of packing solutions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import plotly.graph_objs as go
import plotly.offline as pyo

if TYPE_CHECKING:
    from bin_packer_3d.models.bin import Bin
    from bin_packer_3d.models.placement import Placement, PlacementResult
    from bin_packer_3d.config import VisualizationConfig


class Plotter3D:
    """Interactive 3D visualization of packing results.
    
    Creates Plotly visualizations with:
    - Semi-transparent box meshes
    - Wireframe edges
    - Color coding by box type
    - Hover information with box details
    
    Example:
        >>> plotter = Plotter3D()
        >>> plotter.plot_result(result, output_dir="output/")
    """

    # Color palette for box types
    COLOR_PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    ]

    def __init__(self, config: "VisualizationConfig | None" = None) -> None:
        """Initialize plotter with configuration."""
        from bin_packer_3d.config import VisualizationConfig
        self.config = config or VisualizationConfig()
        self._type_colors: dict[str, str] = {}
        self._next_color_idx = 0

    def _get_color(self, box_type: str) -> str:
        """Get consistent color for a box type."""
        if box_type not in self._type_colors:
            color = self.COLOR_PALETTE[self._next_color_idx % len(self.COLOR_PALETTE)]
            self._type_colors[box_type] = color
            self._next_color_idx += 1
        return self._type_colors[box_type]

    def _make_wireframe(
        self,
        x0: float, x1: float,
        y0: float, y1: float,
        z0: float, z1: float,
        color: str = "black",
        name: str = "wire",
    ) -> go.Scatter3d:
        """Create wireframe edges for a box."""
        corners = [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ]
        edges = [
            (0,1), (1,2), (2,3), (3,0),  # bottom
            (4,5), (5,6), (6,7), (7,4),  # top
            (0,4), (1,5), (2,6), (3,7),  # vertical
        ]
        
        x_vals, y_vals, z_vals = [], [], []
        for start, end in edges:
            x_vals.extend([corners[start][0], corners[end][0], None])
            y_vals.extend([corners[start][1], corners[end][1], None])
            z_vals.extend([corners[start][2], corners[end][2], None])
        
        return go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode="lines",
            line=dict(color=color, width=2),
            name=name,
            hoverinfo="none",
        )

    def _make_mesh(
        self,
        x0: float, x1: float,
        y0: float, y1: float,
        z0: float, z1: float,
        color: str,
        hover_text: str,
    ) -> go.Mesh3d:
        """Create semi-transparent mesh for a box."""
        vertices = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        
        faces = [
            (0,1,2), (0,2,3),  # bottom
            (4,5,6), (4,6,7),  # top
            (0,1,5), (0,5,4),  # front
            (3,2,6), (3,6,7),  # back
            (0,3,7), (0,7,4),  # left
            (1,2,6), (1,6,5),  # right
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
        bin_obj: "Bin",
        title: str = "3D Bin Packing",
    ) -> go.Figure:
        """Create 3D plot for a single bin.
        
        Args:
            bin_obj: Bin with placements to visualize.
            title: Plot title.
        
        Returns:
            Plotly Figure object.
        """
        traces = []
        
        # Bin boundary wireframe
        bin_wire = self._make_wireframe(
            0, bin_obj.length,
            0, bin_obj.width,
            0, bin_obj.height,
            color="black",
            name="Bin",
        )
        traces.append(bin_wire)
        
        # Draw each box
        for placement in bin_obj.placements:
            box = placement.box
            color = self._get_color(box.box_type or box.id)
            
            hover_text = (
                f"ID: {box.id}<br>"
                f"Type: {box.box_type}<br>"
                f"Desc: {box.description}<br>"
                f"Qty: {box.quantity}"
            )
            
            # Mesh
            mesh = self._make_mesh(
                placement.x0, placement.x1,
                placement.y0, placement.y1,
                placement.z0, placement.z1,
                color=color,
                hover_text=hover_text,
            )
            traces.append(mesh)
            
            # Wireframe
            if self.config.show_wireframe:
                wire = self._make_wireframe(
                    placement.x0, placement.x1,
                    placement.y0, placement.y1,
                    placement.z0, placement.z1,
                    color=color,
                    name=f"Box {box.id}",
                )
                traces.append(wire)
        
        layout = go.Layout(
            title=f"{title}<br>Utilization: {bin_obj.utilization_percent:.1f}%",
            scene=dict(
                xaxis=dict(title="Length (mm)"),
                yaxis=dict(title="Width (mm)"),
                zaxis=dict(title="Height (mm)"),
                aspectmode="manual",
                aspectratio=dict(x=1.2, y=1, z=1),
            ),
        )
        
        return go.Figure(data=traces, layout=layout)

    def plot_result(
        self,
        result: "PlacementResult",
        output_dir: Path | str = "output",
    ) -> list[Path]:
        """Create visualizations for all bins in a result.
        
        Args:
            result: Packing result to visualize.
            output_dir: Directory for output files.
        
        Returns:
            List of paths to created HTML files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        created_files: list[Path] = []
        
        for bin_obj in result.bins:
            fig = self.plot_bin(
                bin_obj,
                title=f"Bin {bin_obj.id} - {result.algorithm}",
            )
            
            filename = output_path / f"bin_{bin_obj.id}.html"
            pyo.plot(fig, filename=str(filename), auto_open=self.config.auto_open)
            created_files.append(filename)
            print(f"Saved visualization: {filename}")
        
        return created_files
