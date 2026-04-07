from dataclasses import dataclass

@dataclass
class Config:

    # =========================
    # Training Hyperparameters
    # =========================

    batch_size: int = 64
    epochs: int = 200
    lr: float = 3e-4
    max_len: int = 100

    beta: float = 1.0

    # =========================
    # Model Architecture
    # =========================

    emb_dim: int = 128

    h_dim: int = 128

    z_dim: int = 64


    # =========================
    # System / Logging
    # =========================

    device: str = "cuda"
    project: str = "smiles-compression"
