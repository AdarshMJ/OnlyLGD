from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def plot_graph(
    data: Data,
    title: Optional[str] = None,
    node_color: Optional[str] = None,
    label_map: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)

    label_values = None
    if hasattr(data, "logits") and data.logits is not None:
        label_values = data.logits.argmax(dim=-1).detach().cpu().numpy()
    elif hasattr(data, "y") and data.y is not None:
        label_values = data.y.detach().cpu().numpy()

    if node_color is None and label_values is not None:
        node_color = label_values

    cmap = plt.get_cmap("tab10")
    nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=cmap, ax=ax, linewidths=0.5, edgecolors="black")
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)

    if label_values is not None:
        node_labels = {}
        for node_id, lbl in zip(G.nodes(), label_values):
            mapped = label_map.get(int(lbl), lbl) if label_map else lbl
            node_labels[node_id] = str(mapped)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=6, ax=ax)

    ax.set_axis_off()
    return ax


def save_graph_grid(graphs: list[Data], path: str, cols: int = 4, label_map: Optional[dict] = None) -> None:
    rows = (len(graphs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    for idx, (graph, ax) in enumerate(zip(graphs, axes)):
        plot_graph(graph, label_map=label_map, ax=ax)
    for ax in axes[len(graphs):]:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
