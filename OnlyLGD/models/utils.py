from __future__ import annotations

import importlib
import math
from inspect import isfunction
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import degree, to_dense_adj


def pyg_softmax(src: torch.Tensor, index: torch.Tensor, num_nodes: Optional[int] = None) -> torch.Tensor:
    out = src.exp()
    denom = scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1.0
    return out / denom


def cumsum_pad0(num: torch.Tensor) -> torch.Tensor:
    ret = torch.empty_like(num)
    ret[0] = 0
    ret[1:] = torch.cumsum(num[:-1], dim=0)
    return ret


def num2batch(num_node: torch.Tensor) -> torch.Tensor:
    offset = cumsum_pad0(num_node)
    batch_idx = torch.zeros((offset[-1] + num_node[-1]), device=offset.device, dtype=offset.dtype)
    batch_idx[offset] = 1
    batch_idx[0] = 0
    batch_idx = batch_idx.cumsum_(dim=0)
    return batch_idx


def get_log_deg(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    deg = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.float32)
    return torch.log1p(deg).view(num_nodes, 1)


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    assert timesteps.dim() == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def noise_like(shape: torch.Size, device: torch.device, repeat: bool = False) -> torch.Tensor:
    if repeat:
        noise = torch.randn((1, *shape[1:]), device=device)
        return noise.repeat(shape[0], *((1,) * (len(shape) - 1)))
    return torch.randn(shape, device=device)


def extract_into_tensor(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def extract_into_sparse_tensor(a: torch.Tensor, t: torch.Tensor, num_node: torch.Tensor) -> torch.Tensor:
    out = a.gather(-1, t)
    if num_node.shape[0] == t.shape[0]:
        idx = torch.cat([num2batch(num_node), num2batch(num_node ** 2)], dim=0)
    else:
        idx = num_node
    return out[idx].unsqueeze(1)


def make_beta_schedule(schedule: str, n_timestep: int, linear_start: float = 1e-4, linear_end: float = 2e-2, cosine_s: float = 8e-3) -> np.ndarray:
    if schedule == "linear":
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    elif schedule == "cosine":
        timesteps = (torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s)
        alphas = torch.cos((timesteps / (1 + cosine_s)) * math.pi / 2) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = torch.clamp(betas, 0.0, 0.999)
    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64).sqrt()
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")
    return betas.numpy()


def exists(x) -> bool:
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model: nn.Module, verbose: bool = False) -> int:
    total = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total * 1e-6:.2f}M parameters")
    return total


def symmetrize_tensor(tensor: torch.Tensor, offset: int = 1, scale: torch.Tensor = torch.sqrt(torch.tensor(2.0))) -> torch.Tensor:
    B, N, _, d = tensor.shape
    i, j = torch.triu_indices(N, N, offset=offset)
    tensor[:, i, j, :] += tensor[:, j, i, :]
    tensor[:, i, j, :] /= scale
    tensor[:, j, i, :] = tensor[:, i, j, :]
    return tensor


def symmetrize(edge_index: torch.Tensor, batch: Optional[torch.Tensor], tensor: torch.Tensor, offset: int = 1, scale: torch.Tensor = torch.sqrt(torch.tensor(2.0))) -> torch.Tensor:
    A = to_dense_adj(edge_index, batch, tensor)
    A = symmetrize_tensor(A, offset, scale)
    A = A.reshape(-1, A.shape[-1])
    mask = A.any(dim=1)
    symmetrized = A[mask]
    assert symmetrized.shape[0] == edge_index.shape[1]
    return symmetrized


def instantiate_from_config(config: dict):
    if "target" not in config:
        raise KeyError("Expected key 'target' to instantiate")
    module, cls = config["target"].rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)(**config.get("params", {}))
