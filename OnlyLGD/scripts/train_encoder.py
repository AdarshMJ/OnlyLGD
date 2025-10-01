from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from OnlyLGD.data.synthetic_dataset import SyntheticDatasetConfig, create_dataloaders
from OnlyLGD.models.autoencoder import AutoencoderConfig, GraphAutoencoder
from OnlyLGD.utils.graph import prepare_complete_graph_batch


@dataclass
class EncoderTrainingConfig:
    dataset: SyntheticDatasetConfig
    output_dir: str
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    hidden_dim: int = 256
    latent_dim: int = 128
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    attn_dropout: float = 0.1
    edge_loss_weight: float = 1.0
    cls_loss_weight: float = 1.0
    device: str = "cuda"
    seed: int = 42
    resume: Optional[str] = None
    grad_clip: float = 1.0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the OnlyLGD encoder/decoder on synthetic graphs")
    parser.add_argument("--data-root", type=str, default="data", help="Root directory containing the .pkl graphs")
    parser.add_argument("--dataset-name", type=str, default="featurehomophily", help="Dataset prefix for the synthetic graphs")
    parser.add_argument("--output-dir", type=str, default="outputs/encoder", help="Where to store checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    parser.add_argument("--edge-loss-weight", type=float, default=1.0)
    parser.add_argument("--cls-loss-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--stats-cache", type=str, default=None, help="Optional JSON file to cache condition normalization stats")
    return parser


def parse_config(args: argparse.Namespace) -> EncoderTrainingConfig:
    stats_cache = args.stats_cache or os.path.join(args.output_dir, "condition_stats.json")
    dataset_cfg = SyntheticDatasetConfig(
        root=args.data_root,
        dataset_name=args.dataset_name,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stats_cache=stats_cache,
    )
    return EncoderTrainingConfig(
        dataset=dataset_cfg,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        edge_loss_weight=args.edge_loss_weight,
        cls_loss_weight=args.cls_loss_weight,
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


def infer_num_classes(dataset) -> int:
    sample = dataset[0]
    if hasattr(sample, "num_classes"):
        return int(sample.num_classes)
    labels = sample.y.view(-1)
    return int(labels.max().item() + 1)


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    targets = targets.view(-1).long()
    if mask is not None:
        mask = mask.view(-1).bool()
        if mask.sum() == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        logits = logits[mask]
        targets = targets[mask]
    if targets.numel() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits, targets)


def masked_classification_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> Tuple[int, int]:
    targets = targets.view(-1).long()
    if mask is not None:
        mask = mask.view(-1).bool()
        if mask.sum() == 0:
            return 0, 0
        logits = logits[mask]
        targets = targets[mask]
    if targets.numel() == 0:
        return 0, 0
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    return correct, targets.numel()


def build_autoencoder_config(config: EncoderTrainingConfig, dataset_sample) -> AutoencoderConfig:
    node_in_dim = dataset_sample.x.size(-1)
    edge_in_dim = 1  # dense adjacency targets
    return AutoencoderConfig(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        attn_dropout=config.attn_dropout,
        activation="gelu",
        use_layer_norm=True,
        use_batch_norm=False,
        edge_update=True,
        residual=True,
    )


def train_epoch(
    autoencoder: GraphAutoencoder,
    classifier: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    device: torch.device,
    cfg: EncoderTrainingConfig,
) -> Dict[str, float]:
    autoencoder.train()
    classifier.train()
    total_loss = 0.0
    total_node = 0.0
    total_edge = 0.0
    total_cls = 0.0
    total_correct = 0
    total_count = 0
    steps = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        batch, _, _ = prepare_complete_graph_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        node_latent, edge_latent = autoencoder.encode(batch)
        node_rec, edge_rec = autoencoder.decode(node_latent, edge_latent)

        node_loss = F.mse_loss(node_rec, batch.x)
        edge_targets = batch.edge_attr.view(-1)
        pos_weight = (edge_targets.numel() - edge_targets.sum()) / edge_targets.sum().clamp_min(1.0)
        if edge_rec is None:
            raise RuntimeError("Edge reconstruction output is None; ensure edge features are provided")
        edge_loss = F.binary_cross_entropy_with_logits(edge_rec.view(-1), edge_targets, pos_weight=pos_weight)

        logits = classifier(node_latent)
        train_mask = getattr(batch, "train_mask", None)
        targets = batch.y
        cls_loss = masked_cross_entropy(logits, targets, train_mask)

        loss = node_loss + cfg.edge_loss_weight * edge_loss + cfg.cls_loss_weight * cls_loss
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(list(autoencoder.parameters()) + list(classifier.parameters()), cfg.grad_clip)
        optimizer.step()

        correct, count = masked_classification_stats(logits.detach(), targets.detach(), train_mask)
        total_correct += correct
        total_count += count

        total_loss += loss.item()
        total_node += node_loss.item()
        total_edge += edge_loss.item()
        total_cls += cls_loss.item()
        steps += 1

        pbar.set_postfix({
            "loss": total_loss / steps,
            "node": total_node / steps,
            "edge": total_edge / steps,
            "cls": total_cls / max(steps, 1),
        })

    metrics = {
        "loss": total_loss / max(steps, 1),
        "node_loss": total_node / max(steps, 1),
        "edge_loss": total_edge / max(steps, 1),
        "cls_loss": total_cls / max(steps, 1),
    }
    if total_count > 0:
        metrics["train_acc"] = total_correct / total_count
    return metrics


@torch.no_grad()
def evaluate(
    autoencoder: GraphAutoencoder,
    classifier: nn.Module,
    loader,
    device: torch.device,
    split: str,
    cfg: EncoderTrainingConfig,
) -> Dict[str, float]:
    autoencoder.eval()
    classifier.eval()
    total_loss = 0.0
    total_node = 0.0
    total_edge = 0.0
    total_cls = 0.0
    total_correct = 0
    total_count = 0
    steps = 0

    for batch in loader:
        batch, _, _ = prepare_complete_graph_batch(batch, device)
        node_latent, edge_latent = autoencoder.encode(batch)
        node_rec, edge_rec = autoencoder.decode(node_latent, edge_latent)

        node_loss = F.mse_loss(node_rec, batch.x)
        edge_targets = batch.edge_attr.view(-1)
        pos_weight = (edge_targets.numel() - edge_targets.sum()) / edge_targets.sum().clamp_min(1.0)
        if edge_rec is None:
            raise RuntimeError("Edge reconstruction output is None during evaluation")
        edge_loss = F.binary_cross_entropy_with_logits(edge_rec.view(-1), edge_targets, pos_weight=pos_weight)

        logits = classifier(node_latent)
        mask_attr = f"{split}_mask"
        mask = getattr(batch, mask_attr, None)
        targets = batch.y
        cls_loss = masked_cross_entropy(logits, targets, mask)

        loss = node_loss + cfg.edge_loss_weight * edge_loss + cfg.cls_loss_weight * cls_loss

        correct, count = masked_classification_stats(logits, targets, mask)
        total_correct += correct
        total_count += count

        total_loss += loss.item()
        total_node += node_loss.item()
        total_edge += edge_loss.item()
        total_cls += cls_loss.item()
        steps += 1

    metrics = {
        "loss": total_loss / max(steps, 1),
        "node_loss": total_node / max(steps, 1),
        "edge_loss": total_edge / max(steps, 1),
        "cls_loss": total_cls / max(steps, 1),
    }
    if total_count > 0:
        metrics[f"{split}_acc"] = total_correct / total_count
    return metrics


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    config = parse_config(args)

    ensure_output_dir(config.output_dir)
    if config.dataset.stats_cache:
        Path(config.dataset.stats_cache).parent.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = create_dataloaders(config.dataset, config.batch_size)

    dataset_sample = train_loader.dataset[0]
    auto_cfg = build_autoencoder_config(config, dataset_sample)

    autoencoder = GraphAutoencoder(auto_cfg).to(device)
    num_classes = infer_num_classes(train_loader.dataset)
    classifier = nn.Linear(auto_cfg.latent_dim, num_classes).to(device)

    optimizer = AdamW(
        list(autoencoder.parameters()) + list(classifier.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    config_path = os.path.join(config.output_dir, "encoder_config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)

    history = []
    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    cond_mean, cond_std = train_loader.dataset.condition_stats

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(autoencoder, classifier, optimizer, train_loader, device, config)
        val_metrics = evaluate(autoencoder, classifier, val_loader, device, "val", config)
        scheduler.step()

        epoch_record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(epoch_record)

        val_loss = val_metrics.get("loss", float("inf"))
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            best_state = {
                "autoencoder": autoencoder.state_dict(),
                "classifier": classifier.state_dict(),
            }
            torch.save(
                {
                    "autoencoder_state_dict": autoencoder.state_dict(),
                    "classifier_state_dict": classifier.state_dict(),
                    "autoencoder_config": asdict(auto_cfg),
                    "num_classes": num_classes,
                    "condition_mean": cond_mean.cpu().tolist(),
                    "condition_std": cond_std.cpu().tolist(),
                    "dataset_config": asdict(config.dataset),
                    "training_config": asdict(config),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                os.path.join(config.output_dir, "encoder_best.pt"),
            )

        print(
            f"Epoch {epoch:04d} | train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | best {best_val:.4f}"
        )

    history_path = os.path.join(config.output_dir, "encoder_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    if best_state is not None:
        autoencoder.load_state_dict(best_state["autoencoder"])
        classifier.load_state_dict(best_state["classifier"])

    test_metrics = evaluate(autoencoder, classifier, test_loader, device, "test", config)
    print(
        f"Test loss {test_metrics['loss']:.4f} | "
        f"test node {test_metrics['node_loss']:.4f} | "
        f"test edge {test_metrics['edge_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
