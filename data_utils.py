import torch
import os
from torch.utils.data import Dataset
from datasets import load_dataset

if not os.path.exists("moses_smiles.txt"): # Get dataset if not already present
    ds = load_dataset("antoinebcx/smiles-molecules-moses") # source: https://huggingface.co/datasets/antoinebcx/smiles-molecules-moses?utm_source=chatgpt.com&library=datasets
    with open("moses_smiles.txt", "w") as f:
        for s in ds["train"]["smiles"]:
            f.write(s + "\n")


PAD_TOKEN = "<pad>"      # Padding token (fills unused sequence positions)
SOS_TOKEN = "<sos>"      # Start-of-sequence token
EOS_TOKEN = "<eos>"     # End-of-sequence token


def build_vocab(smiles_list):
    """
    Builds a character-level vocabulary from a list of SMILES strings.
    """
    chars = set("".join(smiles_list))          # Get all unique characters across all SMILES
    vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + sorted(chars)  # Final vocabulary list (special tokens first)

    stoi = {ch: i for i, ch in enumerate(vocab)}   # String-to-index mapping (char -> integer ID)

    itos = {i: ch for ch, i in stoi.items()}   # Index-to-string mapping (integer ID -> char)

    return stoi, itos    # Return both mappings


def encode_smiles(smile, stoi, max_len):
    """
    Converts a single SMILES string into a fixed-length tensor of token IDs.
    """
    tokens = [SOS_TOKEN] + list(smile) + [EOS_TOKEN]  # Add start and end tokens to the SMILES

    tokens = tokens[:max_len]   # Truncate if sequence is too long

    tokens += [PAD_TOKEN] * (max_len - len(tokens))  # Pad sequence to max_len if too short

    return torch.tensor([stoi[t] for t in tokens]) # Convert tokens to indices and return tensor


class SmilesDataset(Dataset):
    """
    PyTorch Dataset for loading and encoding SMILES strings from a file.
    """
    def __init__(self, path, stoi, max_len=100):
        with open(path) as f:   # Open the SMILES text file
            self.smiles = [line.strip() for line in f if line.strip()]  # Read each line as a SMILES string

        self.stoi = stoi         # Store the vocabulary mapping
        self.max_len = max_len   # Maximum sequence length

    def __len__(self):
        return len(self.smiles)      # Total number of SMILES strings

    def __getitem__(self, idx):
        return encode_smiles(
            self.smiles[idx],        # SMILES string at index idx
            self.stoi,               # Vocabulary mapping
            self.max_len             # Fixed sequence length
        )