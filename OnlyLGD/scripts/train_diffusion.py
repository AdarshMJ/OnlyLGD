from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from OnlyLGD.data.synthetic_dataset import SyntheticDatasetConfig, create_dataloaders
from OnlyLGD.models.autoencoder import AutoencoderConfig, GraphAutoencoder
from OnlyLGD.models.denoiser import DenoiserConfig, DenoisingTransformer
from OnlyLGD.models.diffusion import DiffusionConfig, LatentDDPM, LatentGraphBatch
from OnlyLGD.utils.checkpoints import load_encoder_checkpoint
from OnlyLGD.utils.graph import prepare_complete_graph_batch


@dataclass
class DiffusionTrainingConfig:
    dataset: SyntheticDatasetConfig
    encoder_checkpoint: str
    output_dir: str
    epochs: int = 400
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 0.0
    timesteps: int = 1000
    beta_schedule: str = "linear"
    hidden_dim: int = 256
    num_layers: int = 12
    num_heads: int = 8
    dropout: float = 0.1
    use_ema: bool = True
    parameterization: str = "x0"
    device: str = "cuda"
    seed: int = 42
    resume: Optional[str] = None
    grad_clip: float = 1.0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the latent diffusion model on synthetic graphs")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--dataset-name", type=str, default="featurehomophily")
    parser.add_argument("--encoder-checkpoint", type=str, required=True, help="Path to the trained encoder checkpoint")
    parser.add_argument("--output-dir", type=str, default="outputs/diffusion")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", type=str, default="linear", choices=["linear", "cosine", "sqrt", "sqrt_linear"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--parameterization", type=str, default="x0", choices=["x0", "eps"])
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--stats-cache", type=str, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser


def parse_config(args: argparse.Namespace) -> DiffusionTrainingConfig:
    dataset_cfg = SyntheticDatasetConfig(
        root=args.data_root,
        dataset_name=args.dataset_name,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stats_cache=args.stats_cache,
    )
    return DiffusionTrainingConfig(
        dataset=dataset_cfg,
        encoder_checkpoint=args.encoder_checkpoint,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_ema=args.use_ema,
        parameterization=args.parameterization,
        device=args.device,
        seed=args.seed,
        resume=args.resume,
        grad_clip=args.grad_clip,
    )


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_latent_batch(batch, node_latent: torch.Tensor, edge_latent: Optional[torch.Tensor]) -> LatentGraphBatch:
    condition = batch.condition
    if not isinstance(condition, torch.Tensor):
        raise TypeError("Batch condition must be a tensor")
    if condition.dim() == 1:
        condition = condition.view(1, -1)
    num_graphs = batch.num_graphs
    condition = condition.view(num_graphs, -1).to(node_latent.device)

    return LatentGraphBatch(
        node_latent=node_latent,
        edge_latent=edge_latent,
        edge_index=batch.edge_index,
        batch_index=batch.batch,
        condition=condition,
    )


def diffusion_train_epoch(
    diffusion: LatentDDPM,
    autoencoder: GraphAutoencoder,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    cfg: DiffusionTrainingConfig,
) -> Dict[str, float]:
    diffusion.train()
    total_loss = 0.0
    total_node = 0.0
    total_edge = 0.0
    steps = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        batch, _, _ = prepare_complete_graph_batch(batch, device)
        with torch.no_grad():
            node_latent, edge_latent = autoencoder.encode(batch)
        latent_batch = build_latent_batch(batch, node_latent.detach(), edge_latent.detach() if edge_latent is not None else None)

        optimizer.zero_grad(set_to_none=True)
        loss, loss_dict = diffusion(latent_batch)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(diffusion.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss += loss_dict["loss"].item()
        total_node += loss_dict["loss_node"].item()
        total_edge += loss_dict["loss_edge"].item()
        steps += 1

        pbar.set_postfix({
            "loss": total_loss / steps,
            "node": total_node / steps,
            "edge": total_edge / max(steps, 1),
        })

    return {
        "loss": total_loss / max(steps, 1),
        "loss_node": total_node / max(steps, 1),
        "loss_edge": total_edge / max(steps, 1),
    }


@torch.no_grad()
def diffusion_evaluate(
    diffusion: LatentDDPM,
    autoencoder: GraphAutoencoder,
    loader,
    device: torch.device,
    cfg: DiffusionTrainingConfig,
) -> Dict[str, float]:
    diffusion.eval()
    total_loss = 0.0
    total_node = 0.0
    total_edge = 0.0
    steps = 0

    for batch in loader:
        batch, _, _ = prepare_complete_graph_batch(batch, device)
        node_latent, edge_latent = autoencoder.encode(batch)
        latent_batch = build_latent_batch(batch, node_latent, edge_latent)
        loss, loss_dict = diffusion(latent_batch)
        total_loss += loss_dict["loss"].item()
        total_node += loss_dict["loss_node"].item()
        total_edge += loss_dict["loss_edge"].item()
        steps += 1

    return {
        "loss": total_loss / max(steps, 1),
        "loss_node": total_node / max(steps, 1),
        "loss_edge": total_edge / max(steps, 1),
    }


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    config = parse_config(args)

    ensure_output_dir(config.output_dir)
    set_seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    encoder_ckpt = load_encoder_checkpoint(config.encoder_checkpoint, device)

    # Align dataset normalization with encoder training
    encoder_dataset_cfg = encoder_ckpt.get("dataset_config", {})
    if config.dataset.stats_cache is None:
        stats_cache = encoder_dataset_cfg.get("stats_cache")
        if stats_cache:
            config.dataset.stats_cache = stats_cache
        else:
            config.dataset.stats_cache = os.path.join(config.output_dir, "condition_stats.json")

    if config.dataset.stats_cache:
        Path(config.dataset.stats_cache).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(config.dataset.stats_cache):
            with open(config.dataset.stats_cache, "w") as f:
                json.dump({
                    "mean": encoder_ckpt["condition_mean"],
                    "std": encoder_ckpt["condition_std"],
                }, f)

    train_loader, val_loader, _ = create_dataloaders(config.dataset, config.batch_size)

    config_path = os.path.join(config.output_dir, "diffusion_config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)

    auto_cfg = AutoencoderConfig(**encoder_ckpt["autoencoder_config"])
    autoencoder = GraphAutoencoder(auto_cfg).to(device)
    autoencoder.load_state_dict(encoder_ckpt["autoencoder_state_dict"])
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad_(False)

    condition_dim = train_loader.dataset[0].condition.numel()
    denoiser_cfg = DenoiserConfig(
        latent_dim=auto_cfg.latent_dim,
        condition_dim=condition_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        attn_dropout=config.dropout,
        time_embedding_dim=config.hidden_dim,
    )
    denoiser = DenoisingTransformer(denoiser_cfg).to(device)

    diffusion_cfg = DiffusionConfig(
        timesteps=config.timesteps,
        beta_schedule=config.beta_schedule,
        parameterization=config.parameterization,
        use_ema=config.use_ema,
        time_embedding_dim=config.hidden_dim,
    )
    diffusion = LatentDDPM(diffusion_cfg, denoiser).to(device)

    optimizer = AdamW(diffusion.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    history = []
    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, config.epochs + 1):
        train_metrics = diffusion_train_epoch(diffusion, autoencoder, train_loader, device, optimizer, config)
        val_metrics = diffusion_evaluate(diffusion, autoencoder, val_loader, device, config)
        scheduler.step()

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        val_loss = val_metrics.get("loss", float("inf"))
        if val_loss < best_val:
            best_val = val_loss
            best_state = diffusion.state_dict()
            torch.save(
                {
                    "diffusion_state_dict": diffusion.state_dict(),
                    "denoiser_state_dict": denoiser.state_dict(),
                    "diffusion_config": asdict(diffusion_cfg),
                    "denoiser_config": asdict(denoiser_cfg),
                    "encoder_checkpoint": config.encoder_checkpoint,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                os.path.join(config.output_dir, "diffusion_best.pt"),
            )

        print(
            f"Epoch {epoch:04d} | train {train_metrics['loss']:.4f} | "
            f"val {val_metrics['loss']:.4f} | best {best_val:.4f}"
        )

    history_path = os.path.join(config.output_dir, "diffusion_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    if best_state is not None:
        diffusion.load_state_dict(best_state)

    val_metrics = diffusion_evaluate(diffusion, autoencoder, val_loader, device, config)
    print(
        f"Validation loss {val_metrics['loss']:.4f} | "
        f"node {val_metrics['loss_node']:.4f} | "
        f"edge {val_metrics['loss_edge']:.4f}"
    )


if __name__ == "__main__":
    main()
