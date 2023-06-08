from typing import List, Optional, Tuple
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import pandas as pd

COLOR_DF = pd.read_csv('./data/jmol_colors.csv')
# COLOR_DF = None

def get_color_of_atom(atom_symbol: str):
    color = np.array([
        float(COLOR_DF[COLOR_DF['atom'] == atom_symbol]['R'].values[0]) / 255,
        float(COLOR_DF[COLOR_DF['atom'] == atom_symbol]['G'].values[0]) / 255,
        float(COLOR_DF[COLOR_DF['atom'] == atom_symbol]['B'].values[0]) / 255
    ])
    return color

def plot_networkx_mol_graph(
    G: nx.Graph,
    positions: Optional[np.array] = None,
    breaking_bonds: Optional[List[Tuple[int]]] = None,
    forming_bonds: Optional[List[Tuple[int]]] = None,
) -> None:
    if positions is None:
        positions = nx.get_node_attributes(G, "cartesian")
        x, y, z = zip(*positions.values())
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