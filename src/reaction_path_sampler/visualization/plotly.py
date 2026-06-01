import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

_COLOR_DF = None


def _color_df() -> pd.DataFrame:
    """Lazily load the jmol colour table.

    Loaded on first use rather than at import time so that importing this
    module (and everything that transitively imports it) does not require the
    ``./data/jmol_colors.csv`` file to be present relative to the cwd.
    """
    global _COLOR_DF
    if _COLOR_DF is None:
        _COLOR_DF = pd.read_csv("./data/jmol_colors.csv")
    return _COLOR_DF


def get_color_of_atom(atom_symbol: str):
    color_df = _color_df()
    color = np.array(
        [
            float(color_df[color_df["atom"] == atom_symbol]["R"].values[0]) / 255,
            float(color_df[color_df["atom"] == atom_symbol]["G"].values[0]) / 255,
            float(color_df[color_df["atom"] == atom_symbol]["B"].values[0]) / 255,
        ]
    )
    return color


def plot_networkx_mol_graph(
    G: nx.Graph,
    positions: np.ndarray | None = None,
    breaking_bonds: list[tuple[int]] | None = None,
    forming_bonds: list[tuple[int]] | None = None,
) -> None:
    if positions is None:
        positions = nx.get_node_attributes(G, "cartesian")
        x, y, z = zip(*positions.values(), strict=False)
    else:
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    colors = [get_color_of_atom(n[1]["atom_label"]) for n in G.nodes(data=True)]

    # plot nodes
    node_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(size=6, color=colors),
    )

    edge_x, edge_y, edge_z = [], [], []
    for edge in G.edges():
        x0, y0, z0 = positions[edge[0]]
        x1, y1, z1 = positions[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(color="black", width=5),
    )

    fig = go.Figure(data=[node_trace, edge_trace])
    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(title="X"),
    #         yaxis=dict(title="Y"),
    #         zaxis=dict(title="Z"),
    #     ),
    #     showlegend=False,
    # )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.show()
