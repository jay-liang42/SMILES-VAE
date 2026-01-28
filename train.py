import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Import the VAE model
from model import SmilesVAE

# Import dataset utilities from your data file
from data_utils import SmilesDataset, build_vocab

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Uses GPU if available, otherwise falls back to CPU

BATCH_SIZE = 64          # Number of molecules per training batch
EPOCHS = 20              # Number of full passes through the dataset
LR = 3e-4                # Learning rate for the Adam optimizer
MAX_LEN = 100            # Maximum SMILES sequence length
Z_DIM = 32               # Dimensionality of latent space
BETA = 1.0               # Weight on KL divergence (beta-VAE control)


with open("moses_smiles.txt") as f:   # Load the SMILES file
    smiles = [line.strip() for line in f if line.strip()] # Read and clean SMILES strings

# Build vocabulary mappings from the dataset
stoi, itos = build_vocab(smiles)

# Create PyTorch Dataset
dataset = SmilesDataset(
    "moses_smiles.txt",   # Path to SMILES file
    stoi,                 # Vocabulary mapping
    MAX_LEN               # Fixed sequence length
)

# Wrap dataset in a DataLoader for batching and shuffling
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Model, optimizer, loss
model = SmilesVAE(
    vocab_size=len(stoi), # Number of tokens in vocabulary
    z_dim=Z_DIM           # Latent dimension
).to(DEVICE)

optimizer = optim.Adam(
    model.parameters(),   # Parameters to optimize
    lr=LR                 # Learning rate
)

criterion = nn.CrossEntropyLoss(
    ignore_index=stoi["<pad>"],  # Ignore padding tokens in loss
    reduction="sum"              # Sum loss over tokens (standard for VAEs)
)


# Training loop
for epoch in range(EPOCHS):
    model.train()                  # Set model to training mode
    total_loss = 0                 # Accumulate total loss for epoch

    for x in loader:
        x = x.to(DEVICE)           # Move batch to GPU/CPU

        logits, mu, logvar = model(x)
        # logits: (batch, seq_len, vocab_size)
        # mu/logvar: (batch, z_dim)
        
        # Reconstruction loss
        recon_loss = criterion(
            logits.view(-1, logits.size(-1)),  # (batch*seq_len, vocab_size)
            x.view(-1)                         # (batch*seq_len)
        )

        # KL divergence loss
        kl = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        # Total VAE loss
        loss = recon_loss + BETA * kl

        optimizer.zero_grad()       # Clear previous gradients
        loss.backward()             # Backpropagate
        optimizer.step()            # Update parameters

        total_loss += loss.item()   # Accumulate batch loss

    # Normalize loss by number of molecules
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset):.4f}")