import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl  

from data_utils import SmilesDataset, build_vocab, PAD_TOKEN


class SMILESDataModule(pl.LightningDataModule):
    def __init__(self, smiles_file, batch_size, max_len):
        super().__init__()
        self.smiles_file = smiles_file
        self.batch_size = batch_size
        self.max_len = max_len

        # will be set after setup
        self.stoi = None
        self.itos = None
        self.input_dim = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        # Load SMILES strings
        with open(self.smiles_file) as f:
            smiles = [line.strip() for line in f if line.strip()]

        # Build vocabulary
        self.stoi, self.itos = build_vocab(smiles)
        self.input_dim = len(self.stoi) * self.max_len  # flattened one-hot length

        # Create dataset
        dataset = SmilesDataset(
            self.smiles_file,
            self.stoi,
            self.max_len
        )

        # Split 90/10
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

    def _one_hot_flattened(self, batch):
        """
        Convert a batch of sequences (LongTensor) to flattened one-hot FloatTensor.
        Input: list of [seq_len] tensors
        Output: [batch_size, seq_len * vocab_size]
        """
        # batch: list of sequences, stack into tensor
        x = torch.stack(batch, dim=0)  # [batch_size, seq_len]
        batch_size, seq_len = x.shape
        vocab_size = len(self.stoi)

        # One-hot encoding: [batch, seq_len, vocab_size]
        x_oh = torch.nn.functional.one_hot(x, num_classes=vocab_size).float()

        # Flatten sequence dimension: [batch, seq_len * vocab_size]
        x_flat = x_oh.view(batch_size, seq_len * vocab_size)
        return x_flat

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            collate_fn=self._one_hot_flattened
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            collate_fn=self._one_hot_flattened
        )
