import torch
from torch import nn

class SmilesVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for SMILES sequences.
    Uses GRU encoder/decoder with latent bottleneck.
    """
    def __init__(self, vocab_size, emb_dim=128, h_dim=256, z_dim=16, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx

        # --------------------
        # Encoder
        # --------------------
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.encoder_rnn = nn.GRU(emb_dim, h_dim, batch_first=True)
        self.fc_mu = nn.Linear(h_dim, z_dim)      # Latent mean
        self.fc_logvar = nn.Linear(h_dim, z_dim)  # Latent log-variance

        # --------------------
        # Decoder
        # --------------------
        self.fc_z = nn.Linear(z_dim, h_dim)       # Map latent z -> initial hidden
        self.decoder_rnn = nn.GRU(emb_dim, h_dim, batch_first=True)
        self.fc_out = nn.Linear(h_dim, vocab_size)  # Output logits over vocab

    def encode(self, x):
        """Encode input tensor x into latent mean and log-variance."""
        emb = self.embedding(x)                   # (batch, seq_len, emb_dim)
        _, h = self.encoder_rnn(emb)              # h: (1, batch, h_dim)
        h = h.squeeze(0)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample latent vector z from N(mu, sigma^2)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x):
        """Decode latent z using teacher forcing with input x."""
        h0 = self.fc_z(z).unsqueeze(0)            # Initial hidden state for decoder
        emb = self.embedding(x)
        out, _ = self.decoder_rnn(emb, h0)
        return self.fc_out(out)

    def forward(self, x):
        """
        Full forward pass: encode -> reparameterize -> decode.
        Returns:
            logits: token logits
            mu: latent mean
            logvar: latent logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, x)
        return logits, mu, logvar

    def generate(self, z, stoi, itos, max_len=100):
        """
        Generate SMILES string from latent vector z without teacher forcing.
        Autoregressively predicts next token until <eos> or max_len.
        """
        self.eval()
        device = z.device
        with torch.no_grad():
            x = torch.tensor([[stoi["<sos>"]]], device=device)  # Start token
            output = []
            h = self.fc_z(z).unsqueeze(0)

            for _ in range(max_len):
                emb = self.embedding(x)
                out, h = self.decoder_rnn(emb, h)
                logits = self.fc_out(out[:, -1, :])
                token = logits.argmax(dim=-1)
                if token.item() == stoi["<eos>"]:
                    break
                output.append(token.item())
                x = token.unsqueeze(0)

            # Convert token IDs to string, removing special tokens
            return "".join([itos[i] for i in output if i not in (stoi["<sos>"], stoi["<pad>"])])
