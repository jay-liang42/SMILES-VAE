import torch
import torch.nn.functional as F
import random
import pytorch_lightning as pl  

from model import SmilesVAE
from metrics import is_valid_smiles_strict, smiles_similarity
from data_utils import PAD_TOKEN


class SMILESVAE(pl.LightningModule): 
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters()

        self.model = None
        self.reference_pool = None

    def setup(self, stage=None):
        dm = self.trainer.datamodule

        self.stoi = dm.stoi
        self.itos = dm.itos

        with open("moses_smiles.txt") as f:
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

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("recon_loss", recon_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("kl_loss", kl, on_step=False, on_epoch=True, sync_dist=True)

        return loss

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

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        valid_count = 0
        similarities = []

        for _ in range(50):
            z = torch.randn(1, self.cfg.z_dim).to(self.device)
            gen = self.model.generate(z, self.stoi, self.itos)

            if is_valid_smiles_strict(gen):
                valid_count += 1
                ref = random.choice(self.reference_pool)
                sim = smiles_similarity(gen, ref)
                similarities.append(sim)
            else:
                similarities.append(0.0)

        self.log("validity_rate", valid_count / 50, prog_bar=True, sync_dist=True)
        self.log("similarity", sum(similarities) / len(similarities), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.lr
        )
