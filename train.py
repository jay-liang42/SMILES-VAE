import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from model import SmilesVAE
from data_utils import SmilesDataset, build_vocab, PAD_TOKEN
from config import Config
from logger import get_logger

# -----------------------
# Setup
# -----------------------
cfg = Config()
logger = get_logger()
DEVICE = cfg.device if torch.cuda.is_available() else "cpu"

# Initialize W&B experiment logging
wandb.init(project=cfg.project, config=vars(cfg))
logger.info("Starting training")
logger.info(cfg)

# -----------------------
# Prepare Data
# -----------------------
with open("moses_smiles.txt") as f:
    smiles = [line.strip() for line in f if line.strip()]

stoi, itos = build_vocab(smiles)
dataset = SmilesDataset("moses_smiles.txt", stoi, cfg.max_len)
loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

# -----------------------
# Initialize Model
# -----------------------
model = SmilesVAE(
    vocab_size=len(stoi),
    emb_dim=cfg.emb_dim,
    h_dim=cfg.h_dim,
    z_dim=cfg.z_dim,
    pad_idx=stoi[PAD_TOKEN]
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
criterion = nn.CrossEntropyLoss(ignore_index=stoi[PAD_TOKEN], reduction="sum")

# -----------------------
# Training Loop
# -----------------------
for epoch in range(cfg.epochs):
    model.train()
    total_loss = total_kl = total_recon = 0

    for x in tqdm(loader):
        x = x.to(DEVICE)

        # Teacher forcing: decoder input excludes last token, target excludes first token
        decoder_input = x[:, :-1]
        target = x[:, 1:]

        # Forward pass
        logits, mu, logvar = model(decoder_input)

        # Reconstruction loss (cross-entropy)
        recon_loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        # KL divergence (latent regularization)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total VAE loss
        loss = recon_loss + cfg.beta * kl

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_kl += kl.item()
        total_recon += recon_loss.item()

    # Normalize by dataset size for reporting
    epoch_loss = total_loss / len(dataset)
    logger.info(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    # Log metrics to W&B
    wandb.log({
        "epoch": epoch,
        "loss": epoch_loss,
        "kl": total_kl / len(dataset),
        "recon": total_recon / len(dataset),
    })

# -----------------------
# Sampling from Latent Space
# -----------------------
model.eval()
samples = []
with torch.no_grad():
    for _ in range(5):
        z = torch.randn(1, cfg.z_dim).to(DEVICE)
        s = model.generate(z, stoi, itos)
        samples.append(s)
        logger.info(f"Sample: {s}")

wandb.log({"samples": samples})
wandb.finish()
