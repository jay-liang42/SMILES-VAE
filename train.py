import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import is_valid_smiles_strict, smiles_similarity
import random
import wandb
import json
import argparse

from model import SmilesVAE
from data_utils import SmilesDataset, build_vocab, PAD_TOKEN
from config import Config
from logger import get_logger

# -----------------------
# Setup
# -----------------------

cfg = Config()

# -----------------------
# Parse CLI Overrides
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--z_dim", type=int, default=None)
parser.add_argument("--h_dim", type=int, default=None)
parser.add_argument("--emb_dim", type=int, default=None)
parser.add_argument("--beta", type=float, default=None)
parser.add_argument("--lr", type=float, default=None)

args = parser.parse_args()

# Override config values if provided
if args.z_dim is not None:
    cfg.z_dim = args.z_dim
if args.h_dim is not None:
    cfg.h_dim = args.h_dim
if args.emb_dim is not None:
    cfg.emb_dim = args.emb_dim
if args.beta is not None:
    cfg.beta = args.beta
if args.lr is not None:
    cfg.lr = args.lr

logger = get_logger()
DEVICE = cfg.device if torch.cuda.is_available() else "cpu"

# Initialize W&B experiment logging
run_name = (
    f"z{cfg.z_dim}_h{cfg.h_dim}_emb{cfg.emb_dim}"
    f"_beta{cfg.beta}_lr{cfg.lr}"
)

wandb.login(key=json.load(open('/root/gurusmart/wandb_key.json', 'r'))['key'])

wandb.init(
    project=cfg.project,
    name=run_name,
    config=vars(cfg),
)

logger.info("Starting training")
logger.info(cfg)

# -----------------------
# Prepare Data
# -----------------------
with open("moses_smiles.txt") as f:
    smiles = [line.strip() for line in f if line.strip()]
    reference_pool = random.sample(smiles, 1000)

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
# Early Stopping
# -----------------------
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.005):
        self.patience = patience
        self.min_delta = min_delta

        self.best_val_loss = float("inf")
        self.best_similarity = 0.0

        self.counter = 0

    def step(self, val_loss, similarity, model):
        loss_improved = self.best_val_loss - val_loss > self.min_delta
        sim_improved = similarity > self.best_similarity + 1e-4

        if loss_improved or sim_improved:
            if loss_improved:
                self.best_val_loss = val_loss
            if sim_improved:
                self.best_similarity = similarity

            self.counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            print(f"New best | Val Loss: {val_loss:.4f} | Sim: {similarity:.4f}")
            return False
        else:
            self.counter += 1
            print(f"No improvement ({self.counter}/{self.patience})")

            return self.counter >= self.patience

early_stopper = EarlyStopping(patience=20)

# -----------------------
# Training Loop
# -----------------------
for epoch in range(cfg.epochs):
    model.train()
    total_loss = total_kl = total_recon = 0

    for x in tqdm(loader):
        x = x.to(DEVICE)
        decoder_input = x[:, :-1]
        target = x[:, 1:]
        logits, mu, logvar = model(decoder_input)
        logvar = torch.clamp(logvar, -10, 10)
        recon_loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1)
        )
        kl = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        loss = recon_loss + cfg.beta * kl
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        total_kl += kl.item()
        total_recon += recon_loss.item()
    epoch_loss = total_loss / len(dataset)
    logger.info(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
    wandb.log({
        "epoch": epoch,
        "loss": epoch_loss,
        "kl": total_kl / len(dataset),
        "recon": total_recon / len(dataset),
    })

    # -----------------------
    # Validity + Similarity Metrics
    # -----------------------
    model.eval()

    valid_count = 0
    similarities = []

    with torch.no_grad():
        for _ in range(50):
            z = torch.randn(1, cfg.z_dim).to(DEVICE)
            gen = model.generate(z, stoi, itos)

            if is_valid_smiles_strict(gen):
                valid_count += 1
                ref = random.choice(reference_pool)
                sim = smiles_similarity(gen, ref)
                similarities.append(sim)
            else:
                similarities.append(0.0)

    validity_rate = valid_count / 50
    avg_similarity = sum(similarities) / len(similarities)

    logger.info(
        f"Validity: {validity_rate:.3f} | Similarity: {avg_similarity:.3f}"
    )

    wandb.log({
        "validity_rate": validity_rate,
        "similarity": avg_similarity,
    })

    # -----------------------
    # Early Stop Check
    # -----------------------
    if early_stopper.step(epoch_loss, avg_similarity, model):
        logger.info(f"Early stopping triggered at epoch {epoch+1}")
        break

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
