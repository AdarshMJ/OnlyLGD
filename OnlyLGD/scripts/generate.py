from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from OnlyLGD.data.synthetic_dataset import SyntheticDatasetConfig, load_synthetic_dataset
from OnlyLGD.metrics.homophily import compute_homophily_triplet
from OnlyLGD.models.autoencoder import AutoencoderConfig, GraphAutoencoder
from OnlyLGD.models.denoiser import DenoiserConfig, DenoisingTransformer
from OnlyLGD.models.diffusion import DiffusionConfig, LatentDDPM, LatentGraphBatch
from OnlyLGD.utils.checkpoints import load_diffusion_checkpoint, load_encoder_checkpoint
from OnlyLGD.utils.graph import (
    assemble_generated_graphs,
    build_complete_graph_indices,
    node_batch_index_from_counts,
)
from OnlyLGD.visualization.graph import save_graph_grid


@dataclass
class GenerationConfig:
    encoder_checkpoint: str
    diffusion_checkpoint: str
    output_dir: str
    target_feature_h: float = 0.3
    num_graphs: int = 16
    conditioning_strength: float = 1.0
    edge_threshold: float = 0.5
    stochastic_edges: bool = False
    device: str = "cuda"
    seed: int = 42
    reference_dataset: Optional[str] = None
    stats_cache: Optional[str] = None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate graphs using the trained OnlyLGD diffusion model")
    parser.add_argument("--encoder-checkpoint", type=str, required=True)
    parser.add_argument("--diffusion-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/generation")
    parser.add_argument("--target-feature-h", type=float, default=0.3)
    parser.add_argument("--num-graphs", type=int, default=16)
    parser.add_argument("--conditioning-strength", type=float, default=1.0)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--stochastic-edges", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-dataset", type=str, default=None)
    parser.add_argument("--stats-cache", type=str, default=None)
    return parser


def parse_config(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        encoder_checkpoint=args.encoder_checkpoint,
        diffusion_checkpoint=args.diffusion_checkpoint,
        output_dir=args.output_dir,
        target_feature_h=args.target_feature_h,
        num_graphs=args.num_graphs,
        conditioning_strength=args.conditioning_strength,
        edge_threshold=args.edge_threshold,
        stochastic_edges=args.stochastic_edges,
        device=args.device,
        seed=args.seed,
        reference_dataset=args.reference_dataset,
        stats_cache=args.stats_cache,
    )


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_reference_graphs(
    dataset_cfg: SyntheticDatasetConfig,
    num_graphs: int,
) -> List:
    datasets = load_synthetic_dataset(dataset_cfg)
    reference = datasets.get("test") or next(iter(datasets.values()))
    if len(reference) == 0:
        raise ValueError("Reference dataset is empty; cannot generate graphs")
    indices = torch.randint(0, len(reference), (num_graphs,))
    return [reference[int(idx)] for idx in indices]


def build_condition_tensor(
    cond_mean: torch.Tensor,
    cond_std: torch.Tensor,
    target_h: float,
    strength: float,
    num_graphs: int,
    device: torch.device,
) -> torch.Tensor:
    cond_std = cond_std.clamp_min(1e-6)
    raw = cond_mean.clone()
    raw[-2] = target_h
    raw[-1] = target_h
    normalized = (raw - cond_mean) / cond_std
    normalized = normalized * strength
    return normalized.unsqueeze(0).repeat(num_graphs, 1).to(device)


def decode_and_package(
    autoencoder: GraphAutoencoder,
    classifier: nn.Module,
    node_latent: torch.Tensor,
    edge_latent: torch.Tensor,
    counts: List[int],
    num_classes: int,
    edge_threshold: float,
    stochastic_edges: bool,
) -> Dict[str, List]:
    node_rec, edge_rec = autoencoder.decode(node_latent, edge_latent)
    node_logits = classifier(node_latent)
    graphs = assemble_generated_graphs(
        node_rec,
        node_latent,
        edge_rec,
        counts,
        node_logits,
        num_classes,
        threshold=edge_threshold,
        stochastic_edges=stochastic_edges,
    )
    return {
        "graphs": graphs,
        "node_features": node_rec.detach(),
        "edge_logits": edge_rec.detach(),
        "node_logits": node_logits.detach(),
    }


def evaluate_graphs(graphs: List, num_classes: int, show_progress: bool = False) -> List[Dict[str, float]]:
    metrics = []
    iterator = tqdm(graphs, desc="Evaluating homophily", leave=False) if show_progress else graphs
    for graph in iterator:
        label_h, structural_h, feature_h = compute_homophily_triplet(graph, num_classes)
        metrics.append(
            {
                "label_homophily": float(label_h),
                "structural_homophily": float(structural_h),
                "feature_homophily": float(feature_h),
            }
        )
    return metrics


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    config = parse_config(args)
    ensure_output_dir(config.output_dir)
    set_seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    print("Loading checkpoints...")
    encoder_ckpt = load_encoder_checkpoint(config.encoder_checkpoint, device)
    diffusion_ckpt = load_diffusion_checkpoint(config.diffusion_checkpoint, device)

    auto_cfg = AutoencoderConfig(**encoder_ckpt["autoencoder_config"])
    autoencoder = GraphAutoencoder(auto_cfg).to(device)
    autoencoder.load_state_dict(encoder_ckpt["autoencoder_state_dict"])
    autoencoder.eval()

    classifier = nn.Linear(auto_cfg.latent_dim, encoder_ckpt["num_classes"]).to(device)
    classifier.load_state_dict(encoder_ckpt["classifier_state_dict"])
    classifier.eval()

    denoiser_cfg = DenoiserConfig(**diffusion_ckpt["denoiser_config"])
    denoiser = DenoisingTransformer(denoiser_cfg).to(device)
    denoiser.load_state_dict(diffusion_ckpt["denoiser_state_dict"])
    denoiser.eval()

    diffusion_cfg = DiffusionConfig(**diffusion_ckpt["diffusion_config"])
    diffusion = LatentDDPM(diffusion_cfg, denoiser).to(device)
    diffusion.load_state_dict(diffusion_ckpt["diffusion_state_dict"])
    diffusion.eval()

    cond_mean = torch.tensor(encoder_ckpt["condition_mean"], device=device, dtype=torch.float32)
    cond_std = torch.tensor(encoder_ckpt["condition_std"], device=device, dtype=torch.float32)

    dataset_cfg_dict = encoder_ckpt.get("dataset_config", {})
    dataset_cfg = SyntheticDatasetConfig(**dataset_cfg_dict)
    if config.reference_dataset:
        dataset_cfg.root = config.reference_dataset
    if config.stats_cache:
        dataset_cfg.stats_cache = config.stats_cache

    print(f"Preparing {config.num_graphs} initial graphs from dataset '{dataset_cfg.dataset_name}'...")
    reference_graphs = select_reference_graphs(dataset_cfg, config.num_graphs)
    counts = [graph.num_nodes for graph in reference_graphs]
    total_nodes = sum(counts)

    condition = build_condition_tensor(
        cond_mean,
        cond_std,
        config.target_feature_h,
        config.conditioning_strength,
        len(counts),
        device,
    )

    print("Assembling latent tensors...")
    node_latent = torch.zeros((total_nodes, auto_cfg.latent_dim), device=device)
    edge_index, _ = build_complete_graph_indices(counts, device)
    edge_latent = torch.zeros((edge_index.size(1), auto_cfg.latent_dim), device=device)
    batch_index = node_batch_index_from_counts(counts, device)

    latent_batch = LatentGraphBatch(
        node_latent=node_latent,
        edge_latent=edge_latent,
        edge_index=edge_index,
        batch_index=batch_index,
        condition=condition,
    )

    print(f"Running diffusion sampling for {diffusion.cfg.timesteps} timesteps...")
    sampled_node_latent, sampled_edge_latent = diffusion.sample(
        latent_batch,
        show_progress=True,
        desc="Diffusion sampling",
    )
    print("Decoding generated latents...")
    decoded = decode_and_package(
        autoencoder,
        classifier,
        sampled_node_latent,
        sampled_edge_latent,
        counts,
        encoder_ckpt["num_classes"],
        config.edge_threshold,
        config.stochastic_edges,
    )

    graphs = decoded["graphs"]

    cond_raw = condition.detach().cpu() * cond_std.detach().cpu() + cond_mean.detach().cpu()
    for idx, graph in enumerate(graphs):
        graph.condition = cond_raw[idx]
        graph.target_feature_h = torch.tensor(config.target_feature_h, dtype=torch.float32)

    print("Evaluating homophily metrics...")
    metrics = evaluate_graphs(graphs, encoder_ckpt["num_classes"], show_progress=True)
    for graph, metric in zip(graphs, metrics):
        graph.generated_feature_h = torch.tensor(metric["feature_homophily"], dtype=torch.float32)

    print("Saving outputs...")
    config_path = os.path.join(config.output_dir, "generation_config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)

    graphs_path = os.path.join(config.output_dir, "generated_graphs.pt")
    torch.save(graphs, graphs_path)

    summary = {
        "target_feature_h": config.target_feature_h,
        "conditioning_strength": config.conditioning_strength,
        "num_graphs": len(graphs),
        "edge_threshold": config.edge_threshold,
        "stochastic_edges": config.stochastic_edges,
        "metrics": metrics,
    }

    feature_values = [m["feature_homophily"] for m in metrics]
    if feature_values:
        summary["feature_mean"] = float(sum(feature_values) / len(feature_values))
        summary["feature_diff_mean"] = float(
            sum(abs(v - config.target_feature_h) for v in feature_values) / len(feature_values)
        )

    metrics_path = os.path.join(config.output_dir, "generation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    grid_path = os.path.join(config.output_dir, "generated_graphs.png")
    save_graph_grid(graphs[: min(16, len(graphs))], grid_path)

    print(f"Saved {len(graphs)} graphs to {graphs_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Graph grid saved to {grid_path}")


if __name__ == "__main__":
    main()
