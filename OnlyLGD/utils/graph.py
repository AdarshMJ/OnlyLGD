from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse


def node_counts_from_ptr(ptr: Tensor) -> Sequence[int]:
    return (ptr[1:] - ptr[:-1]).tolist()


def build_complete_graph_indices(counts: Sequence[int], device: torch.device) -> Tuple[Tensor, Tensor]:
    edge_indices = []
    edge_batches = []
    node_offset = 0
    for graph_id, num_nodes in enumerate(counts):
        nodes = torch.arange(node_offset, node_offset + num_nodes, device=device)
        row = nodes.repeat_interleave(num_nodes)
        col = nodes.repeat(num_nodes)
        edges = torch.stack([row, col], dim=0)
        edge_indices.append(edges)
        edge_batches.append(torch.full((num_nodes * num_nodes,), graph_id, device=device, dtype=torch.long))
        node_offset += num_nodes
    edge_index = torch.cat(edge_indices, dim=1)
    edge_batch = torch.cat(edge_batches, dim=0)
    return edge_index, edge_batch


def adjacency_targets_from_batch(batch: Batch, counts: Sequence[int], device: torch.device) -> Tensor:
    targets = []
    if hasattr(batch, "A"):
        for graph_id, num_nodes in enumerate(counts):
            adj = batch.A[graph_id, :num_nodes, :num_nodes]
            targets.append(adj.float().reshape(-1, 1))
    else:
        # Fallback: build adjacency from edge_index if dense matrix not available
        start = 0
        for graph_id, num_nodes in enumerate(counts):
            nodes = torch.arange(start, start + num_nodes, device=device)
            mask = torch.isin(batch.edge_index[0], nodes)
            edges = batch.edge_index[:, mask]
            edges = edges - start
            adj = torch.zeros((num_nodes, num_nodes), device=device)
            adj[edges[0], edges[1]] = 1.0
            targets.append(adj.float().reshape(-1, 1))
            start += num_nodes
    return torch.cat(targets, dim=0).to(device)


def prepare_complete_graph_batch(batch: Batch, device: torch.device) -> Tuple[Batch, Sequence[int], Tensor]:
    batch = batch.to(device)
    counts = node_counts_from_ptr(batch.ptr)
    edge_index, edge_batch = build_complete_graph_indices(counts, device)
    edge_attr = adjacency_targets_from_batch(batch, counts, device)
    batch.edge_index = edge_index
    batch.edge_attr = edge_attr
    batch.edge_batch = edge_batch
    return batch, counts, edge_batch


def node_batch_index_from_counts(counts: Sequence[int], device: torch.device) -> Tensor:
    graph_ids = torch.arange(len(counts), device=device)
    repeat = torch.tensor(counts, device=device)
    return graph_ids.repeat_interleave(repeat)


def split_edge_logits(edge_logits: Tensor, counts: Sequence[int]) -> List[Tensor]:
    chunks = []
    offset = 0
    for count in counts:
        num_edges = count * count
        chunk = edge_logits[offset:offset + num_edges]
        chunks.append(chunk.reshape(count, count))
        offset += num_edges
    return chunks


def edge_logits_to_sparse(
    edge_logits: Tensor,
    counts: Sequence[int],
    threshold: float = 0.5,
    stochastic: bool = False,
) -> List[Tensor]:
    adjacency_mats = split_edge_logits(edge_logits.squeeze(-1), counts)
    edge_indices: List[Tensor] = []
    for adj in adjacency_mats:
        probs = torch.sigmoid(adj)
        probs = (probs + probs.T) * 0.5
        probs.fill_diagonal_(0.0)
        if stochastic:
            sampled = torch.bernoulli(probs)
        else:
            sampled = (probs > threshold).float()
        sampled = torch.triu(sampled, diagonal=1)
        sampled = sampled + sampled.T
        edge_index, _ = dense_to_sparse(sampled)
        edge_indices.append(edge_index)
    return edge_indices


def assemble_generated_graphs(
    node_features: Tensor,
    node_latent: Tensor,
    edge_logits: Tensor,
    counts: Sequence[int],
    node_logits: Tensor,
    num_classes: int,
    threshold: float = 0.5,
    stochastic_edges: bool = False,
) -> List[Data]:
    graphs: List[Data] = []
    edge_indices = edge_logits_to_sparse(edge_logits, counts, threshold, stochastic_edges)
    node_offset = 0
    for graph_id, (count, edge_index) in enumerate(zip(counts, edge_indices)):
        node_slice = slice(node_offset, node_offset + count)
        x = node_features[node_slice].detach().cpu()
        z = node_latent[node_slice].detach().cpu()
        logits = node_logits[node_slice].detach().cpu()
        y = logits.argmax(dim=-1)
        data = Data(
            x=x,
            edge_index=edge_index.cpu(),
            y=y,
            logits=logits,
            latent=z,
        )
        data.num_classes = num_classes
        graphs.append(data)
        node_offset += count
    return graphs