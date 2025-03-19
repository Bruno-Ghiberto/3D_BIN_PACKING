# plotter.py

import os
import plotly.graph_objs as go
import plotly.offline as pyo

class HeuristicPlotter:
    """
    This class plots a multi-bin 3D packing solution where each box
    is a semi-transparent prism, color-coded by 'CAJA', with hover info
    from columns: 'BOX_ID', 'CAJA', 'DESCRIPCION', 'CANTIDAD', etc.
    """

    def __init__(self):
        # Simple palette to color boxes by their 'CAJA' type.
        # Feel free to expand or adjust these colors.
        self.color_palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        # Dictionary to map each unique CAJA -> color
        self.type_to_color = {}
        self.next_color_idx = 0

    def get_color_for_type(self, box_type):
        """
        Returns a consistent color for each unique 'box_type' (CAJA).
        If unused, assign the next color in the palette.
        """
        if box_type not in self.type_to_color:
            color = self.color_palette[self.next_color_idx % len(self.color_palette)]
            self.type_to_color[box_type] = color
            self.next_color_idx += 1
        return self.type_to_color[box_type]

    def _make_wireframe(self, x0, x1, y0, y1, z0, z1, color='black', name="wire"):
        """
        Creates a wireframe (edges) of a rectangular prism in 3D using Scatter3d.
        """
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
            (0,1), (1,2), (2,3), (3,0),   # bottom face
            (4,5), (5,6), (6,7), (7,4),   # top face
            (0,4), (1,5), (2,6), (3,7)    # vertical edges
        ]
        x_vals, y_vals, z_vals = [], [], []
        for (start, end) in edges:
            x_vals.extend([corners[start][0], corners[end][0], None])
            y_vals.extend([corners[start][1], corners[end][1], None])
            z_vals.extend([corners[start][2], corners[end][2], None])

        return go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='lines',
            line=dict(color=color, width=2),
            name=name,
            hoverinfo='none'  # wireframe doesn't need hover
        )

    def _make_prism_mesh(self, x0, x1, y0, y1, z0, z1, color, hover_text):
        """
        Creates a semi-transparent rectangular prism (Mesh3d).
        (x0, y0, z0), (x1, y1, z1) define the opposite corners.
        """
        # 8 corner points
        vertices = [
            (x0, y0, z0),  # 0
            (x1, y0, z0),  # 1
            (x1, y1, z0),  # 2
            (x0, y1, z0),  # 3
            (x0, y0, z1),  # 4
            (x1, y0, z1),  # 5
            (x1, y1, z1),  # 6
            (x0, y1, z1),  # 7
        ]
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        zs = [v[2] for v in vertices]

        # Indices of the 12 triangular faces
        faces = [
            # bottom
            (0,1,2), (0,2,3),
            # top
            (4,5,6), (4,6,7),
            # front
            (0,1,5), (0,5,4),
            # back
            (3,2,6), (3,6,7),
            # left
            (0,3,7), (0,7,4),
            # right
            (1,2,6), (1,6,5)
        ]
        i_vals = [f[0] for f in faces]
        j_vals = [f[1] for f in faces]
        k_vals = [f[2] for f in faces]

        mesh = go.Mesh3d(
            x=xs, y=ys, z=zs,
            i=i_vals, j=j_vals, k=k_vals,
            color=color,
            opacity=0.3,         # 30% transparency
            hoverinfo='text',
            text=hover_text,      # display on hover
            name="box"
        )
        return mesh

    def plot_single_bin(
        self, df_bin, bin_length, bin_width, bin_height,
        title="Heuristic 3D Packing"
    ):
        """
        Plots a single bin with boxes from df_bin.
        Expected columns: [BOX_ID, CAJA, DESCRIPCION, CANTIDAD, x0, y0, z0, x1, y1, z1, ...]
        """
        data_traces = []

        # 1) Draw the bin boundary as a wireframe
        bin_wire = self._make_wireframe(
            x0=0, x1=bin_length,
            y0=0, y1=bin_width,
            z0=0, z1=bin_height,
            color='black',
            name='Bin'
        )
        data_traces.append(bin_wire)

        # 2) For each box in this bin, draw a semi-transparent mesh + wireframe
        for i, row in df_bin.iterrows():
            x0, x1 = row['x0'], row['x1']
            y0, y1 = row['y0'], row['y1']
            z0, z1 = row['z0'], row['z1']

            # Extract any data needed for hover
            item_id = row.get('BOX_ID', '???')
            caja_type = row.get('CAJA', '???')  # "Box Type"
            desc = row.get('DESCRIPCION', '???')
            qty = row.get('CANTIDAD', 1)

            hover_txt = (
                f"Item: {item_id}<br>"
                f"Box Type: {caja_type}<br>"
                f"Desc: {desc}<br>"
                f"Qty: {qty}"
            )

            # Color by CAJA
            color = self.get_color_for_type(caja_type)

            # Create the 3D mesh
            mesh_trace = self._make_prism_mesh(
                x0, x1, y0, y1, z0, z1,
                color=color,
                hover_text=hover_txt
            )
            data_traces.append(mesh_trace)

            # Optionally add a wireframe around the box
            box_wire = self._make_wireframe(
                x0, x1, y0, y1, z0, z1,
                color=color,
                name=f"Box {item_id}"
            )
            data_traces.append(box_wire)

        # 3) Build layout
        layout = go.Layout(
            title=title,
            scene=dict(
                xaxis=dict(title='Length (mm)'),
                yaxis=dict(title='Width (mm)'),
                zaxis=dict(title='Height (mm)'),
                aspectmode='manual',
                aspectratio=dict(x=1.2, y=1, z=1),
            ),
        )

        fig = go.Figure(data=data_traces, layout=layout)
        return fig

    def plot_all_bins(
        self, df_placements,
        bin_length, bin_width, bin_height,
        output_folder
    ):
        """
        For each unique BIN_ID in df_placements, create an HTML file.
        df_placements columns must include:
          ['BIN_ID','BOX_ID','x0','y0','z0','x1','y1','z1','CAJA','DESCRIPCION','CANTIDAD',...]
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        unique_bins = df_placements['BIN_ID'].unique()
        for bin_id in unique_bins:
            df_bin = df_placements[df_placements['BIN_ID'] == bin_id].copy()
            title_str = f"Bin {bin_id} - 3D Packing"

            fig = self.plot_single_bin(
                df_bin=df_bin,
                bin_length=bin_length,
                bin_width=bin_width,
                bin_height=bin_height,
                title=title_str
            )

            filename = os.path.join(output_folder, f"bin_{bin_id}.html")
            pyo.plot(fig, filename=filename, auto_open=False)
            print(f"Saved 3D plot for Bin {bin_id} => {filename}")
