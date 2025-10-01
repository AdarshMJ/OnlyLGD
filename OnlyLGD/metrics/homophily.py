from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops


def measure_label_homophily(data: Data) -> float:
    edge_index = data.edge_index
    labels = data.y
    edge_index, _ = remove_self_loops(edge_index)
    if edge_index.numel() == 0:
        return 0.0
    same = (labels[edge_index[0]] == labels[edge_index[1]]).float()
    return float(same.mean().item())


def measure_structural_homophily(data: Data, num_classes: int) -> float:
    edge_index, _ = remove_self_loops(data.edge_index)
    labels = data.y.long()
    num_nodes = data.x.size(0)
    if edge_index.numel() == 0 or num_nodes == 0:
        return 0.5

    device = data.x.device
    adj = torch.zeros((num_nodes, num_nodes), device=device)
    adj[edge_index[0], edge_index[1]] = 1.0

    y_one_hot = F.one_hot(labels, num_classes=num_classes).float().to(device)
    neighbor_dist = adj @ y_one_hot
    row_sums = neighbor_dist.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    neighbor_dist = neighbor_dist / row_sums

    def get_max_std(classes: int) -> float:
        return float(np.sqrt((1 - 1 / classes) / classes)) if classes > 1 else 0.0

    class_scores = []
    for cls in range(num_classes):
        class_nodes = (labels == cls).nonzero(as_tuple=False).flatten()
        if class_nodes.numel() <= 1:
            continue
        class_dist = neighbor_dist[class_nodes]
        if class_dist.size(1) <= 1:
            continue
        std_list = class_dist.std(dim=0)
        std_max = get_max_std(class_dist.size(1))
        if std_max > 0:
            hom_val = (1 - std_list / std_max).mean().item()
            class_scores.append(hom_val)

    if class_scores:
        return float(np.mean(class_scores))
    return 0.5


def spectral_radius_sp_matrix(edge_index: torch.Tensor, values: torch.Tensor, num_nodes: int) -> float:
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla

    if edge_index.shape[0] != 2:
        edge_index = edge_index.t()
    matrix = sp.coo_matrix((values.cpu().numpy(), (edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())), shape=(num_nodes, num_nodes))
    try:
        eigenvalues, _ = sla.eigs(matrix, k=1, which="LM")
        return float(np.abs(eigenvalues[0]).real)
    except Exception:
        counts = torch.bincount(edge_index[0], minlength=num_nodes)
        return float(counts.max().item())


def measure_feature_homophily(data: Data, num_classes: int, max_iterations: int = 50) -> float:
    edge_index, _ = remove_self_loops(data.edge_index)
    labels = data.y.long()
    num_nodes = data.x.size(0)
    if edge_index.numel() == 0 or num_nodes == 0:
        return 0.0

    device = data.x.device
    adj = torch.zeros((num_nodes, num_nodes), device=device)
    adj[edge_index[0], edge_index[1]] = 1.0
    identity = torch.eye(num_nodes, device=device)

    try:
        spectral_radius = spectral_radius_sp_matrix(
            edge_index.cpu(), torch.ones(edge_index.size(1)), num_nodes
        )
    except Exception:
        degrees = torch.bincount(edge_index[0].cpu(), minlength=num_nodes)
        spectral_radius = float(degrees.max().item()) if degrees.numel() > 0 else 0.0

    y_one_hot = F.one_hot(labels, num_classes=num_classes).float().to(device)
    features = data.x.float()

    h_candidates = torch.linspace(-0.9, 0.9, max_iterations)
    best_h = 0.0
    min_error = float("inf")

    for h_value in h_candidates:
        spectral = float(spectral_radius)
        weight = h_value.item() / spectral if spectral > 0 else 0.0
        if abs(weight) >= 0.95:
            continue
        try:
            transformed = (identity - weight * adj) @ features
            class_sums = y_one_hot.t() @ transformed
            class_counts = y_one_hot.sum(dim=0, keepdim=True).t()
            class_counts[class_counts == 0] = 1.0
            class_means = class_sums / class_counts
            reconstructed = y_one_hot @ class_means
            error = torch.abs(reconstructed - transformed).sum().item()
        except Exception:
            continue

        if error < min_error:
            min_error = error
            best_h = h_value.item()

    return best_h


def compute_homophily_triplet(data: Data, num_classes: int) -> Tuple[float, float, float]:
    label_h = measure_label_homophily(data)
    structural_h = measure_structural_homophily(data, num_classes)
    feature_h = measure_feature_homophily(data, num_classes)
    return label_h, structural_h, feature_h
