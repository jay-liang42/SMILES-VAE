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
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
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
            emb_dim=self.cfg.emb_dim,
            h_dim=self.cfg.h_dim,
            z_dim=self.cfg.z_dim,
            pad_idx=self.stoi[PAD_TOKEN]
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch

        decoder_input = x[:, :-1]
        target = x[:, 1:]

        logits, mu, logvar = self.model(decoder_input)

        recon_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=self.stoi[PAD_TOKEN],
            reduction="sum"
        )

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.cfg.beta * kl

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("recon_loss", recon_loss, on_epoch=True)
        self.log("kl_loss", kl, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics.get("train_loss_epoch")

        if loss is not None:
            logger.info(f"Epoch {self.current_epoch} Loss: {loss:.4f}")

        valid_count = 0
        similarities = []

        for _ in range(50):
            z = torch.randn(1, self.cfg.z_dim).to(self.device)
            gen = self.model.generate(z, self.stoi, self.itos)

            if is_valid_smiles_strict(gen):
                valid_count += 1
                ref = random.choice(self.reference_pool)
                similarities.append(smiles_similarity(gen, ref))
            else:
                similarities.append(0.0)

        train_validity = valid_count / 50
        train_similarity = sum(similarities) / len(similarities)

        logger.info(
            f"Train Validity: {train_validity:.3f} | Train Similarity: {train_similarity:.3f}"
        )

        self.model.train()

    def validation_step(self, batch, batch_idx):
        x = batch

        decoder_input = x[:, :-1]
        target = x[:, 1:]

        logits, mu, logvar = self.model(decoder_input)

        recon_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=self.stoi[PAD_TOKEN],
            reduction="sum"
        )

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.cfg.beta * kl

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        valid_count = 0
        similarities = []

        for _ in range(50):
            z = torch.randn(1, self.cfg.z_dim).to(self.device)
            gen = self.model.generate(z, self.stoi, self.itos)

            if is_valid_smiles_strict(gen):
                valid_count += 1
                ref = random.choice(self.reference_pool)
                similarities.append(smiles_similarity(gen, ref))
            else:
                similarities.append(0.0)

        validity = valid_count / 50
        similarity = sum(similarities) / len(similarities)

        self.log("validity_rate", validity, prog_bar=True)
        self.log("similarity", similarity, prog_bar=True)

        logger.info(
            f"Validity: {validity:.3f} | Similarity: {similarity:.3f}"
        )

        self.model.train()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
