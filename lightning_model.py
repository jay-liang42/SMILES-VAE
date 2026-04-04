import torch
import torch.nn as nn
import pytorch_lightning as pl
import logging

logger = logging.getLogger(__name__)


class SMILESVAE(pl.LightningModule):
    def __init__(self, vocab_size, config):
        super().__init__()

        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.h_dim = config["h_dim"]
        self.z_dim = config["z_dim"]
        self.lr = config["lr"]

        # KL annealing params
        self.beta_max = config.get("beta", 1.0)
        self.kl_anneal_epochs = config.get("kl_anneal_epochs", 50)

        self.embedding = nn.Embedding(vocab_size, self.h_dim)

        self.encoder = nn.GRU(self.h_dim, self.h_dim, batch_first=True)

        self.fc_mu = nn.Linear(self.h_dim, self.z_dim)
        self.fc_logvar = nn.Linear(self.h_dim, self.z_dim)

        self.decoder = nn.GRU(self.h_dim + self.z_dim, self.h_dim, batch_first=True)
        self.output = nn.Linear(self.h_dim, vocab_size)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def encode(self, x):
        x = self.embedding(x)
        _, h = self.encoder(x)
        h = h.squeeze(0)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z):
        x = self.embedding(x)
        z = z.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat([x, z], dim=-1)

        out, _ = self.decoder(x)
        return self.output(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(x, z)
        return logits, mu, logvar

    def compute_beta(self):
        # Linear annealing
        if self.current_epoch >= self.kl_anneal_epochs:
            return self.beta_max
        return self.beta_max * (self.current_epoch / self.kl_anneal_epochs)

    def training_step(self, batch, batch_idx):
        x = batch

        logits, mu, logvar = self(x)

        recon_loss = self.loss_fn(
            logits.view(-1, self.vocab_size),
            x.view(-1)
        )

        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        beta = self.compute_beta()

        loss = recon_loss + beta * kl

        self.log("train_loss", loss)
        self.log("recon_loss", recon_loss)
        self.log("kl_loss", kl)
        self.log("beta", beta)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch

        logits, mu, logvar = self(x)

        recon_loss = self.loss_fn(
            logits.view(-1, self.vocab_size),
            x.view(-1)
        )

        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        beta = self.compute_beta()
        loss = recon_loss + beta * kl

        self.log("val_loss", loss)
        self.log("val_kl", kl)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
