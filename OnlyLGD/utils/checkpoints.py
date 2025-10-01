from __future__ import annotations

import os
from typing import Dict

import torch


def _load_checkpoint(path: str, device: torch.device) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dictionary, got {type(checkpoint)!r}")
    return checkpoint


def load_encoder_checkpoint(path: str, device: torch.device) -> Dict:
    checkpoint = _load_checkpoint(path, device)
    required_keys = {
        "autoencoder_state_dict",
        "classifier_state_dict",
        "autoencoder_config",
        "num_classes",
        "condition_mean",
        "condition_std",
    }
    missing = required_keys - checkpoint.keys()
    if missing:
        raise KeyError(f"Encoder checkpoint missing keys: {sorted(missing)}")
    return checkpoint


def load_diffusion_checkpoint(path: str, device: torch.device) -> Dict:
    checkpoint = _load_checkpoint(path, device)
    required_keys = {
        "diffusion_state_dict",
        "denoiser_state_dict",
        "diffusion_config",
        "denoiser_config",
    }
    missing = required_keys - checkpoint.keys()
    if missing:
        raise KeyError(f"Diffusion checkpoint missing keys: {sorted(missing)}")
    return checkpoint
