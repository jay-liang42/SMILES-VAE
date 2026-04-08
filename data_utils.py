import torch
from torch.utils.data import Dataset
import os
from datasets import load_dataset

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

# Download dataset if missing
if not os.path.exists("moses_smiles.txt"):
    ds = load_dataset("antoinebcx/smiles-molecules-moses")
    with open("moses_smiles.txt", "w") as f:
        for s in ds["train"]["smiles"]:
            f.write(s + "\n")

def build_vocab(smiles_list):
    """
    Build character-level vocab from SMILES, adding PAD, SOS, EOS.
    """
    chars = set("".join(smiles_list))
    vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + sorted(chars)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode_smiles(smile, stoi, max_len):
    """
    Encode SMILES into fixed-length tensor of token IDs.
    Unknown characters are ignored to prevent KeyErrors.
    """
    tokens = [SOS_TOKEN] + [ch for ch in smile if ch in stoi] + [EOS_TOKEN]  # ignore unknowns
    tokens = tokens[:max_len]  # truncate if too long
    tokens += [PAD_TOKEN] * (max_len - len(tokens))  # pad if too short
    return torch.tensor([stoi[t] for t in tokens])

class SmilesDataset(Dataset):
    """
    PyTorch Dataset for SMILES strings.
    Can read from a file or take a list of SMILES directly.
    """
    def __init__(self, path=None, smiles_list=None, stoi=None, max_len=100):
        if smiles_list is not None:
            self.smiles = smiles_list
        elif path is not None:
            with open(path) as f:
                self.smiles = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError("Provide either 'path' or 'smiles_list'")
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return encode_smiles(self.smiles[idx], self.stoi, self.max_len)
