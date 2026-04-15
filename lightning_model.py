import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import random
import logging

from model import SmilesVAE
from metrics import is_valid_smiles_strict, smiles_similarity
from data_utils import PAD_TOKEN

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SMILESVAE(pl.LightningModule):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.cfg = config
        self.save_hyperparameters()

        self.model = None
        self.reference_pool = None
        self.stoi = None
        self.itos = None

    def setup(self, stage=None):
        dm = self.trainer.datamodule

        self.stoi = dm.stoi
        self.itos = dm.itos

        with open(dm.smiles_file) as f:
            smiles = [line.strip() for line in f if line.strip()]
            self.reference_pool = random.sample(smiles, 1000)

        self.model = SmilesVAE(
            vocab_size=len(self.stoi),
            emb_dim=self.cfg["emb_dim"],
            h_dim=self.cfg["h_dim"],
            z_dim=self.cfg["z_dim"],
            pad_idx=self.stoi[PAD_TOKEN]
        )

    def forward(self, x):
        """Forward pass using SmilesVAE's internal teacher forcing."""
        logits, mu, logvar, x_target = self.model(x, epoch=self.current_epoch)
        return logits, mu, logvar, x_target

    # -----------------------
    # TRAINING
    # -----------------------
    def training_step(self, batch, batch_idx):
        x = batch

        logits, mu, logvar, x_target = self.forward(x)

        # Reconstruction loss
        recon_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            x_target.reshape(-1),
            ignore_index=self.stoi[PAD_TOKEN],
            reduction="mean"
        )

        # KL divergence
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        free_bits = 0.1
        kl = torch.clamp(kl, min=free_bits)
        kl = kl.mean()

        if self.current_epoch < 10:
            beta = 0.0
        else:
            beta = min(0.05, (self.current_epoch - 10) / 40 * 0.05)
        loss = recon_loss + beta * kl

        # Logging
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("recon_loss", recon_loss, on_epoch=True)
        self.log("kl_loss", kl, on_epoch=True, prog_bar=True)
        self.log("beta", beta, on_epoch=True)
        self.log("mu_std", mu.std(), on_epoch=True, prog_bar=True)
        self.log("mu_mean", mu.mean(), on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        self.model.eval()
    
        samples = []
        valid_count = 0
        similarities = []
    
        with torch.no_grad():
            for _ in range(10):
                z = torch.randn(1, self.cfg["z_dim"], device=self.device)
                gen = self.model.generate(z, self.stoi, self.itos)
                samples.append(gen)
        
                if is_valid_smiles_strict(gen):
                    valid_count += 1
                    ref = random.choice(self.reference_pool)
                    similarities.append(smiles_similarity(gen, ref))
                else:
                    similarities.append(0.0)
    
        logger.info(f"\n===== Epoch {self.current_epoch} Samples =====")
        for s in samples[:5]:
            logger.info(f"{s} | valid={is_valid_smiles_strict(s)}")
    
        loss = self.trainer.callback_metrics.get("train_loss")
        if loss is not None:
            logger.info(f"Epoch {self.current_epoch} Loss: {loss:.4f}")
    
        train_validity = valid_count / 10
        train_similarity = sum(similarities) / len(similarities)
    
        logger.info(
            f"Train Validity: {train_validity:.3f} | Train Similarity: {train_similarity:.3f}"
        )
    
        self.model.train()

    # -----------------------
    # VALIDATION
    # -----------------------
    def validation_step(self, batch, batch_idx):
        x = batch

        logits, mu, logvar, x_target = self.forward(x)

        recon_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            x_target.reshape(-1),
            ignore_index=self.stoi[PAD_TOKEN],
            reduction="mean"
        )

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        free_bits = 0.1
        kl = torch.clamp(kl, min=free_bits)
        kl = kl.mean()

        if self.current_epoch < 10:
            beta = 0.0
        else:
            beta = min(0.05, (self.current_epoch - 10) / 40 * 0.05)
        loss = recon_loss + beta * kl

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_recon_loss", recon_loss)
        self.log("beta", beta, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        valid_count = 0
        similarities = []

        with torch.no_grad():
            for _ in range(10):
                z = torch.randn(1, self.cfg["z_dim"], device=self.device)
                gen = self.model.generate(z, self.stoi, self.itos)
    
                if is_valid_smiles_strict(gen):
                    valid_count += 1
                    ref = random.choice(self.reference_pool)
                    similarities.append(smiles_similarity(gen, ref))
                else:
                    similarities.append(0.0)

        validity = valid_count / 10
        similarity = sum(similarities) / len(similarities)

        self.log("validity_rate", validity, prog_bar=True)
        self.log("similarity", similarity, prog_bar=True)

        logger.info(
            f"Validity: {validity:.3f} | Similarity: {similarity:.3f}"
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg["lr"])
