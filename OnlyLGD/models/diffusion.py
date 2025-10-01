from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

from .denoiser import DenoisingTransformer
from .utils import (
    default,
    extract_into_tensor,
    get_timestep_embedding,
    make_beta_schedule,
    noise_like,
)


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_schedule: str = "linear"
    linear_start: float = 1e-4
    linear_end: float = 2e-2
    cosine_s: float = 8e-3
    parameterization: str = "eps"
    loss_type: str = "l2"
    use_ema: bool = False
    time_embedding_dim: int = 256


@dataclass
class LatentGraphBatch:
    node_latent: Tensor
    edge_latent: Optional[Tensor]
    edge_index: Tensor
    batch_index: Tensor
    condition: Tensor

    def num_graphs(self) -> int:
        return self.condition.size(0)

    def edge_batch_index(self) -> Tensor:
        return self.batch_index[self.edge_index[0]]


class LatentDDPM(nn.Module):
    def __init__(self, cfg: DiffusionConfig, denoiser: DenoisingTransformer) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = denoiser
        self.parameterization = cfg.parameterization
        self.loss_type = cfg.loss_type
        self.register_schedule()

    def register_schedule(self) -> None:
        betas = make_beta_schedule(
            self.cfg.beta_schedule,
            self.cfg.timesteps,
            self.cfg.linear_start,
            self.cfg.linear_end,
            self.cfg.cosine_s,
        )
        betas = torch.tensor(betas, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        noise = default(noise, lambda: torch.randn_like(x_start))
        fac1 = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        fac2 = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return fac1 * x_start + fac2 * noise

    def predict_start_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        fac1 = extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        fac2 = extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return fac1 * x_t - fac2 * noise

    def get_time_embedding(self, t_graph: Tensor) -> Tensor:
        return get_timestep_embedding(t_graph, self.cfg.time_embedding_dim)

    def p_losses(self, batch: LatentGraphBatch, t_graph: Tensor) -> Tuple[Tensor, dict]:
        node_noise = torch.randn_like(batch.node_latent)
        node_t = t_graph[batch.batch_index]
        node_noisy = self.q_sample(batch.node_latent, node_t, node_noise)

        if batch.edge_latent is not None:
            edge_noise = torch.randn_like(batch.edge_latent)
            edge_t = t_graph[batch.edge_batch_index()]
            edge_noisy = self.q_sample(batch.edge_latent, edge_t, edge_noise)
        else:
            edge_noise = None
            edge_noisy = None

        temb = self.get_time_embedding(t_graph)
        pred_node, pred_edge = self.model(
            node_noisy,
            edge_noisy,
            batch.edge_index,
            batch.batch_index,
            batch.condition,
            temb,
        )

        losses = {}
        if self.parameterization == "eps":
            node_target = node_noise
            edge_target = edge_noise
        elif self.parameterization == "x0":
            node_target = batch.node_latent
            edge_target = batch.edge_latent
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")

        node_loss = F.mse_loss(pred_node, node_target)
        losses["loss_node"] = node_loss
        if pred_edge is not None and edge_target is not None:
            edge_loss = F.mse_loss(pred_edge, edge_target)
        else:
            edge_loss = torch.tensor(0.0, device=node_loss.device)
        losses["loss_edge"] = edge_loss
        total_loss = node_loss + edge_loss
        losses["loss"] = total_loss
        return total_loss, losses

    def forward(self, batch: LatentGraphBatch) -> Tuple[Tensor, dict]:
        device = batch.node_latent.device
        t_graph = torch.randint(0, self.cfg.timesteps, (batch.num_graphs(),), device=device, dtype=torch.long)
        return self.p_losses(batch, t_graph)

    @torch.no_grad()
    def p_sample(self, batch: LatentGraphBatch, t: Tensor, node_latent: Tensor, edge_latent: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        temb = self.get_time_embedding(t)
        pred_node, pred_edge = self.model(
            node_latent,
            edge_latent,
            batch.edge_index,
            batch.batch_index,
            batch.condition,
            temb,
        )

        if self.parameterization == "eps":
            node_pred = pred_node
            edge_pred = pred_edge
            x0_node = self.predict_start_from_noise(node_latent, t[batch.batch_index], node_pred)
            if edge_latent is not None and edge_pred is not None:
                x0_edge = self.predict_start_from_noise(edge_latent, t[batch.edge_batch_index()], edge_pred)
            else:
                x0_edge = None
        else:
            x0_node = pred_node
            x0_edge = pred_edge

        coef1 = extract_into_tensor(self.posterior_mean_coef1, t[batch.batch_index], node_latent.shape)
        coef2 = extract_into_tensor(self.posterior_mean_coef2, t[batch.batch_index], node_latent.shape)
        node_mean = coef1 * x0_node + coef2 * node_latent
        log_var = extract_into_tensor(self.posterior_log_variance_clipped, t[batch.batch_index], node_latent.shape)
        noise = noise_like(node_latent.shape, node_latent.device)
        nonzero_mask = (t[batch.batch_index] != 0).float().unsqueeze(-1)
        node_latent = node_mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

        if edge_latent is not None and x0_edge is not None:
            coef1_edge = extract_into_tensor(self.posterior_mean_coef1, t[batch.edge_batch_index()], edge_latent.shape)
            coef2_edge = extract_into_tensor(self.posterior_mean_coef2, t[batch.edge_batch_index()], edge_latent.shape)
            edge_mean = coef1_edge * x0_edge + coef2_edge * edge_latent
            log_var_edge = extract_into_tensor(self.posterior_log_variance_clipped, t[batch.edge_batch_index()], edge_latent.shape)
            edge_noise = noise_like(edge_latent.shape, edge_latent.device)
            nonzero_edge = (t[batch.edge_batch_index()] != 0).float().unsqueeze(-1)
            edge_latent = edge_mean + nonzero_edge * torch.exp(0.5 * log_var_edge) * edge_noise

        return node_latent, edge_latent

    @torch.no_grad()
    def sample(self, batch: LatentGraphBatch, show_progress: bool = False, desc: str = "sampling") -> Tuple[Tensor, Optional[Tensor]]:
        node_latent = torch.randn_like(batch.node_latent)
        edge_latent = torch.randn_like(batch.edge_latent) if batch.edge_latent is not None else None
        timesteps = list(range(self.cfg.timesteps - 1, -1, -1))
        iterator = tqdm(timesteps, desc=desc) if show_progress else timesteps
        device = node_latent.device
        for t_scalar in iterator:
            t_graph = torch.full((batch.num_graphs(),), t_scalar, device=device, dtype=torch.long)
            node_latent, edge_latent = self.p_sample(batch, t_graph, node_latent, edge_latent)
        return node_latent, edge_latent