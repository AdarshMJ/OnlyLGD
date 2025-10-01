from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add
from torch_scatter.composite import scatter_softmax


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, activation: nn.Module, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        activation,
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


@dataclass
class GraphTransformerLayerConfig:
    hidden_dim: int
    edge_dim: int
    num_heads: int
    dropout: float = 0.1
    attn_dropout: float = 0.1
    activation: str = "gelu"
    residual: bool = True
    use_layer_norm: bool = True
    use_batch_norm: bool = False
    edge_update: bool = True


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, cfg: GraphTransformerLayerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        h = cfg.num_heads
        if d % h != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.head_dim = d // h
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.edge_proj = nn.Linear(cfg.edge_dim, d) if cfg.edge_dim > 0 else None
        self.out_proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(cfg.attn_dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        src, dst = edge_index
        q = self.q_proj(x).view(-1, self.cfg.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.cfg.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.cfg.num_heads, self.head_dim)

        q_dst = q[dst]
        k_src = k[src]
        logits = (q_dst * k_src).sum(-1) * self.scale

        if self.edge_proj is not None and edge_attr is not None:
            e = self.edge_proj(edge_attr).view(-1, self.cfg.num_heads, self.head_dim)
            logits = logits + (q_dst * e).sum(-1) * self.scale

        attn = scatter_softmax(logits, dst, dim=0)
        attn = self.dropout(attn)
        messages = attn.unsqueeze(-1) * v[src]
        out = scatter_add(messages, dst, dim=0, dim_size=x.size(0))
        out = out.reshape(x.size(0), -1)
        return self.out_proj(out)


class EdgeUpdate(nn.Module):
    def __init__(self, cfg: GraphTransformerLayerConfig) -> None:
        super().__init__()
        act = _activation(cfg.activation)
        in_dim = 3 * cfg.hidden_dim if cfg.edge_dim == 0 else 2 * cfg.hidden_dim + cfg.edge_dim
        self.mlp = _mlp(in_dim, cfg.hidden_dim, cfg.hidden_dim, act, cfg.dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor]) -> Tensor:
        src, dst = edge_index
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), x.size(1), device=x.device, dtype=x.dtype)
        inp = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        return self.mlp(inp)


class GraphTransformerLayer(nn.Module):
    def __init__(self, cfg: GraphTransformerLayerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.attn = MultiHeadGraphAttention(cfg)
        act = _activation(cfg.activation)
        self.ffn = _mlp(cfg.hidden_dim, cfg.hidden_dim * 2, cfg.hidden_dim, act, cfg.dropout)
        self.edge_update = EdgeUpdate(cfg) if cfg.edge_update else None

        self.dropout = nn.Dropout(cfg.dropout)
        self.use_ln = cfg.use_layer_norm
        self.use_bn = cfg.use_batch_norm
        if self.use_ln:
            self.ln1 = nn.LayerNorm(cfg.hidden_dim)
            self.ln2 = nn.LayerNorm(cfg.hidden_dim)
            if self.edge_update is not None:
                self.ln_edge = nn.LayerNorm(cfg.hidden_dim)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(cfg.hidden_dim)
            self.bn2 = nn.BatchNorm1d(cfg.hidden_dim)
            if self.edge_update is not None:
                self.bn_edge = nn.BatchNorm1d(cfg.hidden_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        temb: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if temb is not None:
            temb_node, temb_edge = temb
            x = x + temb_node
            if edge_attr is not None:
                edge_attr = edge_attr + temb_edge

        res = x
        x = self.attn(x, edge_index, edge_attr)
        x = self.dropout(x)
        if self.cfg.residual:
            x = x + res
        if self.use_ln:
            x = self.ln1(x)
        if self.use_bn:
            x = self.bn1(x)

        res2 = x
        x = self.ffn(x)
        x = self.dropout(x)
        if self.cfg.residual:
            x = x + res2
        if self.use_ln:
            x = self.ln2(x)
        if self.use_bn:
            x = self.bn2(x)

        if self.edge_update is not None:
            res_edge = edge_attr
            edge_attr = self.edge_update(x, edge_index, edge_attr)
            edge_attr = self.dropout(edge_attr)
            if self.cfg.residual and res_edge is not None:
                edge_attr = edge_attr + res_edge
            if self.use_ln:
                edge_attr = self.ln_edge(edge_attr)
            if self.use_bn:
                edge_attr = self.bn_edge(edge_attr)

        return x, edge_attr


@dataclass
class GraphTransformerConfig:
    input_dim: int
    edge_dim: int
    hidden_dim: int
    output_dim: int
    num_heads: int
    num_layers: int
    dropout: float = 0.1
    attn_dropout: float = 0.1
    activation: str = "gelu"
    residual: bool = True
    use_layer_norm: bool = True
    use_batch_norm: bool = False
    edge_update: bool = True
    use_time_embedding: bool = False
    time_embedding_dim: int = 0


class GraphTransformerEncoder(nn.Module):
    def __init__(self, cfg: GraphTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.edge_proj = nn.Linear(cfg.edge_dim, cfg.hidden_dim) if cfg.edge_dim > 0 else None
        layer_cfg = GraphTransformerLayerConfig(
            hidden_dim=cfg.hidden_dim,
            edge_dim=cfg.hidden_dim if cfg.edge_dim > 0 else 0,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            attn_dropout=cfg.attn_dropout,
            activation=cfg.activation,
            residual=cfg.residual,
            use_layer_norm=cfg.use_layer_norm,
            use_batch_norm=cfg.use_batch_norm,
            edge_update=cfg.edge_update,
        )
        self.layers = nn.ModuleList([GraphTransformerLayer(layer_cfg) for _ in range(cfg.num_layers)])
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        self.edge_output_proj = nn.Linear(cfg.hidden_dim, cfg.output_dim) if cfg.edge_update else None

        if cfg.use_time_embedding:
            self.time_mlp = nn.Sequential(
                nn.Linear(cfg.time_embedding_dim, cfg.hidden_dim),
                _activation(cfg.activation),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
        else:
            self.time_mlp = None

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        temb: Optional[Tensor] = None,
        batch_index: Optional[Tensor] = None,
        edge_batch_index: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        h = self.input_proj(x)
        e = self.edge_proj(edge_attr) if (edge_attr is not None and self.edge_proj is not None) else edge_attr

        if self.time_mlp is not None and temb is not None:
            if batch_index is None:
                raise ValueError("batch_index must be provided when using time embeddings")
            time_embed = self.time_mlp(temb)
            temb_node = time_embed[batch_index]
            if edge_batch_index is None:
                edge_batch_index = batch_index[edge_index[0]]
            temb_edge = time_embed[edge_batch_index]
        else:
            temb_node = temb_edge = None

        for layer in self.layers:
            h, e = layer(h, edge_index, e, (temb_node, temb_edge) if temb_node is not None else None)

        h_out = self.output_proj(h)
        e_out = self.edge_output_proj(e) if (self.edge_output_proj is not None and e is not None) else e
        return h_out, e_out


class GraphTransformerDecoder(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim),
        )

    def forward(self, node_latents: Tensor, edge_latents: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        node_out = self.node_mlp(node_latents)
        edge_out = self.edge_mlp(edge_latents) if edge_latents is not None else None
        return node_out, edge_out