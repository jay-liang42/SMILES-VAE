import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from data_utils import SmilesDataset, build_vocab


class SMILESDataModule(pl.LightningDataModule):
    def __init__(self, smiles_file, batch_size, max_len):
        super().__init__()
        self.smiles_file = smiles_file
        self.batch_size = batch_size
        self.max_len = max_len

        self.stoi = None
        self.itos = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        # Load SMILES strings
        with open(self.smiles_file) as f:
            smiles = [line.strip() for line in f if line.strip()]

        # Build vocabulary
        self.stoi, self.itos = build_vocab(smiles)

        # Create dataset
        dataset = SmilesDataset(
            self.smiles_file,
            self.stoi,
            self.max_len
        )

        # Train/validation split (90/10)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True
        )
