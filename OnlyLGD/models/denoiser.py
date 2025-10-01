from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .graph_transformer import GraphTransformerConfig, GraphTransformerEncoder


@dataclass
class DenoiserConfig:
    latent_dim: int
    condition_dim: int
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
    use_time_embedding: bool = True
    time_embedding_dim: int = 256


class DenoisingTransformer(nn.Module):
    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.latent_dim + cfg.condition_dim
        encoder_cfg = GraphTransformerConfig(
            input_dim=input_dim,
            edge_dim=input_dim,
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
            use_time_embedding=cfg.use_time_embedding,
            time_embedding_dim=cfg.time_embedding_dim,
        )
        self.transformer = GraphTransformerEncoder(encoder_cfg)

    def forward(
        self,
        node_latent: Tensor,
        edge_latent: Optional[Tensor],
        edge_index: Tensor,
        batch_index: Tensor,
        condition: Tensor,
        temb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        node_condition = condition[batch_index]
        node_input = torch.cat([node_latent, node_condition], dim=-1)

        if edge_latent is not None:
            edge_batch = batch_index[edge_index[0]]
            edge_condition = condition[edge_batch]
            edge_input = torch.cat([edge_latent, edge_condition], dim=-1)
        else:
            edge_input = None

        edge_batch = batch_index[edge_index[0]]
        node_out, edge_out = self.transformer(
            node_input,
            edge_index,
            edge_input,
            temb,
            batch_index=batch_index,
            edge_batch_index=edge_batch,
        )
        return node_out, edge_out