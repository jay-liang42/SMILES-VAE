import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

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
    parser.add_argument("--project", type=str, default="smiles-vae")
    parser.add_argument("--run_name", type=str, default="run1")

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
    # WandB Logger
    # -----------------------
    wandb_run = wandb.init(project=args.project, name=args.run_name)
    wandb_logger = WandbLogger(experiment=wandb_run)

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
    # Callbacks
    # -----------------------
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="epoch_{epoch:02d}",
        save_last=True
    )

    # -----------------------
    # Trainer
    # -----------------------
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        benchmark=True,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        logger=wandb_logger,
        callbacks=[early_stop, lr_monitor, checkpoint],
        deterministic=False
    )

    # -----------------------
    # Training
    # -----------------------
    trainer.fit(model, datamodule=data)

    # -----------------------
    # Finish WandB run
    # -----------------------
    wandb.finish()


if __name__ == "__main__":
    main()
