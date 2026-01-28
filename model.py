import torch
from torch import nn


class SmilesVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for SMILES strings.
    Uses a GRU encoder and decoder with a continuous latent space.
    """
    def __init__(self, vocab_size, emb_dim=128, h_dim=256, z_dim=32):
        super().__init__()

        # --------------------
        # Encoder components
        # --------------------

        self.embedding = nn.Embedding(
            vocab_size,              # Size of vocabulary (number of unique tokens)
            emb_dim,                 # Dimension of embedding vectors
            padding_idx=0            # PAD_TOKEN index (must be 0 — matches your vocab)
        )

        self.encoder_rnn = nn.GRU(
            emb_dim,                 # Input size = embedding dimension
            h_dim,                   # Hidden state size
            batch_first=True         # Input shape: (batch, seq_len, emb_dim)
        )

        self.fc_mu = nn.Linear(
            h_dim,                   # Encoder hidden state size
            z_dim                    # Latent mean dimension
        )

        self.fc_logvar = nn.Linear(
            h_dim,                   # Encoder hidden state size
            z_dim                    # Latent log-variance dimension
        )

        # --------------------
        # Decoder components
        # --------------------

        self.fc_z = nn.Linear(
            z_dim,                   # Latent vector size
            h_dim                    # Decoder initial hidden state size
        )

        self.decoder_rnn = nn.GRU(
            emb_dim,                 # Input size = embedding dimension
            h_dim,                   # Hidden state size
            batch_first=True         # Input shape: (batch, seq_len, emb_dim)
        )

        self.fc_out = nn.Linear(
            h_dim,                   # Decoder hidden state size
            vocab_size               # Output logits over vocabulary
        )

    def encode(self, x):
        """
        Encode input SMILES into latent mean and log-variance.

        x: Tensor of shape (batch_size, seq_len)
        """
        emb = self.embedding(x)              # (batch, seq_len, emb_dim)

        _, h = self.encoder_rnn(emb)         # h: (1, batch, h_dim)

        h = h.squeeze(0)                     # (batch, h_dim)

        mu = self.fc_mu(h)                   # Latent mean (batch, z_dim)
        logvar = self.fc_logvar(h)            # Latent log-variance (batch, z_dim)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Sample latent vector z using the reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)         # Standard deviation
        eps = torch.randn_like(std)           # Random noise ~ N(0, I)
        return mu + eps * std                 # Sampled latent vector z

    def decode(self, z, x):
        """
        Decode latent vector z into output token logits.

        z: Tensor of shape (batch_size, z_dim)
        x: Input tokens for teacher forcing (batch_size, seq_len)
        """
        h0 = self.fc_z(z).unsqueeze(0)        # Initial hidden state (1, batch, h_dim)

        emb = self.embedding(x)               # (batch, seq_len, emb_dim)

        out, _ = self.decoder_rnn(emb, h0)    # (batch, seq_len, h_dim)

        return self.fc_out(out)               # (batch, seq_len, vocab_size)

    def forward(self, x):
        """
        Full VAE forward pass.

        Returns:
        - logits: unnormalized token scores
        - mu: latent mean
        - logvar: latent log-variance
        """
        mu, logvar = self.encode(x)           # Encode input
        z = self.reparameterize(mu, logvar)   # Sample latent vector
        logits = self.decode(z, x)            # Decode with teacher forcing

        return logits, mu, logvar