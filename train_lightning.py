import argparse
import lightning as L

from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from lightning_model import SMILESVAE
from data_module import SMILESDataModule
from config import Config


def main():

    cfg = Config()

    # -----------------------
    # CLI Overrides
    # -----------------------
    parser = argparse.ArgumentParser()

    parser.add_argument("--z_dim", type=int)
    parser.add_argument("--h_dim", type=int)
    parser.add_argument("--emb_dim", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--lr", type=float)

    args = parser.parse_args()

    if args.z_dim:
        cfg.z_dim = args.z_dim
    if args.h_dim:
        cfg.h_dim = args.h_dim
    if args.emb_dim:
        cfg.emb_dim = args.emb_dim
    if args.beta:
        cfg.beta = args.beta
    if args.lr:
        cfg.lr = args.lr

    # -----------------------
    # Model
    # -----------------------
    model = SMILESVAE(cfg)

    # -----------------------
    # Data
    # -----------------------
    data = SMILESDataModule(
        smiles_file="moses_smiles.txt",
        batch_size=cfg.batch_size,
        max_len=cfg.max_len
    )

    # -----------------------
    # Early Stopping
    # -----------------------
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min"
    )

    # -----------------------
    # Trainer
    # -----------------------
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        benchmark=True,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        callbacks=[early_stop],
        deterministic=False
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    main()
