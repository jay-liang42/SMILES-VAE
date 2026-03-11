import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from lightning_model import SMILESVAE
from data_module import SMILESDataModule
from config import Config


def main():

    cfg = Config()

    model = SMILESVAE(cfg)

    data = SMILESDataModule(
        "moses_smiles.txt",
        batch_size=cfg.batch_size,
        max_len=cfg.max_len
    )

    wandb_logger = WandbLogger(
        project=cfg.project,
        name=f"z{cfg.z_dim}_h{cfg.h_dim}_emb{cfg.emb_dim}_beta{cfg.beta}_lr{cfg.lr}"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min"
    )

    trainer = L.Trainer(
    max_epochs=cfg.epochs,
    accelerator="auto",
    devices=1,
    precision="16-mixed",
    benchmark=True,
    gradient_clip_val=0.5,
    log_every_n_steps=50,
    logger=wandb_logger,
    callbacks=[early_stop],
    deterministic=False
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    main()
