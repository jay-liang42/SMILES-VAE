from dataclasses import dataclass

@dataclass
class Config:

    # =========================
    # Training Hyperparameters
    # =========================

    batch_size: int = 64
    # Number of samples processed before each optimizer step.
    # Larger = faster training but more VRAM usage.

    epochs: int = 5
    # Full passes over the dataset.
    # Increase this to improve convergence.

    lr: float = 3e-4
    # Learning rate for the optimizer.
    # Controls how large parameter updates are each step.

    max_len: int = 100
    # Maximum token length for SMILES sequences.
    # Longer molecules get truncated/padded to this size.

    beta: float = 1.0
    # Weight of KL-divergence term in VAE loss.
    # Higher -> stronger latent compression
    # Lower -> better reconstruction


    # =========================
    # Model Architecture
    # =========================

    emb_dim: int = 128
    # Token embedding dimension.
    # Larger captures richer chemical syntax features.

    h_dim: int = 256
    # Hidden dimension of encoder/decoder layers.
    # Controls model capacity.

    z_dim: int = 16
    # Latent space dimensionality.
    # Smaller = more compression (your main goal)
    # Larger = more expressive latent representation


    # =========================
    # System / Logging
    # =========================

    device: str = "cuda"
    # Compute device:
    #   "cuda" -> GPU
    #   "cpu"  -> fallback if GPU unavailable

    project: str = "smiles-compression"
    # Weights & Biases project name.
    # Used for experiment tracking/logging

