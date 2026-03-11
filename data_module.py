import lightning as L
from torch.utils.data import DataLoader

from data_utils import SmilesDataset, build_vocab
from config import Config


class SMILESDataModule(L.LightningDataModule):

    def __init__(self, smiles_file, batch_size, max_len):
        super().__init__()

        self.smiles_file = smiles_file
        self.batch_size = batch_size
        self.max_len = max_len

    def setup(self, stage=None):

        with open(self.smiles_file) as f:
            smiles = [line.strip() for line in f if line.strip()]

        self.stoi, self.itos = build_vocab(smiles)

        self.dataset = SmilesDataset(
            self.smiles_file,
            self.stoi,
            self.max_len
        )

    def train_dataloader(self):

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
