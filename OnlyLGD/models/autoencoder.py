from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from .graph_transformer import GraphTransformerConfig, GraphTransformerDecoder, GraphTransformerEncoder


@dataclass
class AutoencoderConfig:
    node_in_dim: int
    edge_in_dim: int
    latent_dim: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    dropout: float = 0.1
    attn_dropout: float = 0.1
    activation: str = "gelu"
    residual: bool = True
    use_layer_norm: bool = True
    use_batch_norm: bool = False
    edge_update: bool = True


class GraphAutoencoder(nn.Module):
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        encoder_cfg = GraphTransformerConfig(
            input_dim=cfg.node_in_dim,
            edge_dim=cfg.edge_in_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.latent_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            attn_dropout=cfg.attn_dropout,
            activation=cfg.activation,
            residual=cfg.residual,
            use_layer_norm=cfg.use_layer_norm,
            use_batch_norm=cfg.use_batch_norm,
            edge_update=cfg.edge_update,
        )
        self.encoder = GraphTransformerEncoder(encoder_cfg)
        self.decoder = GraphTransformerDecoder(cfg.latent_dim, cfg.latent_dim, cfg.hidden_dim)
        self.node_reconstruction = nn.Linear(cfg.latent_dim, cfg.node_in_dim)
        self.edge_reconstruction = (
            nn.Linear(cfg.latent_dim, cfg.edge_in_dim) if cfg.edge_in_dim > 0 else None
        )

    def encode(self, data: Data) -> Tuple[Tensor, Optional[Tensor]]:
        node_latent, edge_latent = self.encoder(data.x, data.edge_index, getattr(data, "edge_attr", None))
        return node_latent, edge_latent

    def decode(self, node_latent: Tensor, edge_latent: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        node_decoded, edge_decoded = self.decoder(node_latent, edge_latent)
        node_rec = self.node_reconstruction(node_decoded)
        edge_rec = self.edge_reconstruction(edge_decoded) if (edge_decoded is not None and self.edge_reconstruction is not None) else edge_decoded
        return node_rec, edge_rec

    def graph_latent(self, node_latent: Tensor, batch_index: Tensor) -> Tensor:
        return global_mean_pool(node_latent, batch_index)

    def forward(self, data: Data) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        node_latent, edge_latent = self.encode(data)
        node_rec, edge_rec = self.decode(node_latent, edge_latent)
        graph_latent = self.graph_latent(node_latent, data.batch)
        return node_rec, edge_rec, graph_latent