from __future__ import annotations

import os
import json
import pickle
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


@dataclass
class SyntheticDatasetConfig:
    """Configuration for loading the synthetic homophily graphs."""

    root: str = "data"
    dataset_name: str = "featurehomophily"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    stratify_by_homophily: bool = True
    normalize_conditions: bool = True
    stats_cache: Optional[str] = None

    def validate(self) -> None:
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Synthetic dataset root not found: {self.root}")
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(
                "Train/val/test ratios must sum to 1.0, got "
                f"{self.train_ratio}, {self.val_ratio}, {self.test_ratio}"
            )


class SyntheticHomophilyDataset(Dataset):
    """PyTorch dataset for synthetic graphs with controllable feature homophily."""

    def __init__(
        self,
        graphs: Sequence[Data],
        metadata: pd.DataFrame,
        indices: Sequence[int],
        normalize: bool = True,
        condition_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        self.graphs = graphs
        self.metadata = metadata
        self.indices = list(indices)
        self.normalize = normalize

        if condition_stats is not None:
            self._cond_mean, self._cond_std = condition_stats
        else:
            conds = torch.stack([self._extract_condition(graphs[i]) for i in self.indices])
            self._cond_mean = conds.mean(dim=0)
            self._cond_std = conds.std(dim=0).clamp_min(1e-6)

    @property
    def condition_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._cond_mean, self._cond_std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Data:
        idx = self.indices[item]
        data = self.graphs[idx]
        condition = self._extract_condition(data)
        if self.normalize:
            condition = (condition - self._cond_mean) / self._cond_std

        data = data.clone()
        data.condition = condition
        data.target_feature_h = torch.as_tensor(float(self._get_target_feature_h(data)), dtype=torch.float32)
        data.actual_feature_h = torch.as_tensor(float(self._get_actual_feature_h(data)), dtype=torch.float32)

        return data

    def _extract_condition(self, data: Data) -> torch.Tensor:
        if hasattr(data, "stats"):
            stats = data.stats.view(-1)
        else:
            stats = torch.zeros(18, dtype=torch.float32)

        target = torch.tensor([self._get_target_feature_h(data)], dtype=torch.float32)
        actual = torch.tensor([self._get_actual_feature_h(data)], dtype=torch.float32)
        hom_vector = torch.cat([target, actual])
        return torch.cat([stats, hom_vector]).float()

    @staticmethod
    def _get_target_feature_h(data: Data) -> float:
        value = getattr(data, "feature_homophily", None)
        if value is None:
            return float(getattr(data, "target_feature_hom", 0.5))
        if isinstance(value, torch.Tensor):
            return float(value.item())
        return float(value)

    @staticmethod
    def _get_actual_feature_h(data: Data) -> float:
        if hasattr(data, "stats") and data.stats.numel() >= 18:
            return float(data.stats.view(-1)[-1].item())
        if hasattr(data, "actual_feature_hom"):
            value = data.actual_feature_hom
            if isinstance(value, torch.Tensor):
                return float(value.item())
            return float(value)
        return SyntheticHomophilyDataset._get_target_feature_h(data)


def _load_pickled_graphs(path: str) -> List[Data]:
    with open(path, "rb") as f:
        graphs: List[Data] = pickle.load(f)
    return graphs


def _discover_dataset_files(root: str, dataset_name: str) -> Tuple[str, str, str]:
    base = os.path.join(root, dataset_name)
    graphs_path = os.path.join(root, f"{dataset_name}_graphs.pkl")
    metadata_path = os.path.join(root, f"{dataset_name}_log.csv")
    summary_path = os.path.join(root, f"{dataset_name}_summary.txt")

    if os.path.exists(graphs_path):
        return graphs_path, metadata_path, summary_path

    if os.path.isdir(base):
        graphs_path = os.path.join(base, f"{dataset_name}_graphs.pkl")
        metadata_path = os.path.join(base, f"{dataset_name}_log.csv")
        summary_path = os.path.join(base, f"{dataset_name}_summary.txt")
        if os.path.exists(graphs_path):
            return graphs_path, metadata_path, summary_path

    raise FileNotFoundError(
        "Could not locate synthetic dataset files. Expected either "
        f"{graphs_path} or {os.path.join(base, dataset_name + '_graphs.pkl')}"
    )


def load_synthetic_dataset(config: SyntheticDatasetConfig) -> Dict[str, SyntheticHomophilyDataset]:
    config.validate()
    graphs_path, metadata_path, _ = _discover_dataset_files(config.root, config.dataset_name)

    graphs = _load_pickled_graphs(graphs_path)
    metadata = pd.read_csv(metadata_path) if os.path.exists(metadata_path) else pd.DataFrame()

    indices_map = _build_split_indices(
        len(graphs),
        graphs,
        metadata,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.seed,
        config.stratify_by_homophily,
    )

    condition_stats = None
    if config.stats_cache and os.path.exists(config.stats_cache):
        with open(config.stats_cache, "r") as f:
            cache = json.load(f)
        condition_stats = (
            torch.tensor(cache["mean"], dtype=torch.float32),
            torch.tensor(cache["std"], dtype=torch.float32),
        )

    datasets = {}
    for split, idxs in indices_map.items():
        if split == "train" and condition_stats is None:
            dataset = SyntheticHomophilyDataset(graphs, metadata, idxs, config.normalize_conditions)
            condition_stats = dataset.condition_stats
        else:
            dataset = SyntheticHomophilyDataset(
                graphs,
                metadata,
                idxs,
                config.normalize_conditions,
                condition_stats,
            )
        datasets[split] = dataset

    if config.stats_cache:
        os.makedirs(os.path.dirname(config.stats_cache), exist_ok=True)
        mean, std = condition_stats
        with open(config.stats_cache, "w") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)

    return datasets


def _build_split_indices(
    total_size: int,
    graphs: Sequence[Data],
    metadata: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify: bool,
) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)

    if stratify:
        buckets: Dict[float, List[int]] = {}
        for idx in range(total_size):
            graph = graphs[idx]
            target = SyntheticHomophilyDataset._get_target_feature_h(graph)
            level = float(np.round(target, decimals=2))
            buckets.setdefault(level, []).append(idx)

        train_idx, val_idx, test_idx = [], [], []
        for level, idxs in buckets.items():
            idxs = np.array(idxs)
            rng.shuffle(idxs)
            n = len(idxs)
            n_train = int(np.floor(train_ratio * n))
            n_val = int(np.floor(val_ratio * n))
            train_idx.extend(idxs[:n_train].tolist())
            val_idx.extend(idxs[n_train:n_train + n_val].tolist())
            test_idx.extend(idxs[n_train + n_val :].tolist())
        return {
            "train": sorted(train_idx),
            "val": sorted(val_idx),
            "test": sorted(test_idx),
        }

    indices = np.arange(total_size)
    rng.shuffle(indices)
    n_train = int(np.floor(train_ratio * total_size))
    n_val = int(np.floor(val_ratio * total_size))
    return {
        "train": indices[:n_train].tolist(),
        "val": indices[n_train:n_train + n_val].tolist(),
        "test": indices[n_train + n_val :].tolist(),
    }


def create_dataloaders(
    config: SyntheticDatasetConfig,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    datasets = load_synthetic_dataset(config)
    loaders = {}
    for split, dataset in datasets.items():
        shuffle = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return loaders["train"], loaders["val"], loaders["test"]
