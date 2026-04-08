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

    def decode(self, z, x, teacher_forcing_ratio=0.95):
        h = self.fc_z(z).unsqueeze(0)
        batch_size, seq_len = x.size()
    
        outputs = []
    
        input_token = x[:, 0].unsqueeze(1)  # <sos>
    
        for t in range(seq_len):
            emb = self.embedding(input_token)
            out, h = self.decoder_rnn(emb, h)
            logits = self.fc_out(out)
    
            outputs.append(logits)
    
            pred_token = logits.argmax(dim=-1)
    
            if self.training and t + 1 < seq_len:
                use_teacher = torch.rand(batch_size, device=x.device) < teacher_forcing_ratio
                use_teacher = use_teacher.unsqueeze(1)
    
                next_input = torch.where(
                    use_teacher,
                    x[:, t + 1].unsqueeze(1),
                    pred_token
                )
            else:
                next_input = pred_token
    
            input_token = next_input
    
        return torch.cat(outputs, dim=1)

    def forward(self, x):
        """
        Full forward pass: encode -> reparameterize -> decode.

        Returns:
            logits: token logits aligned with x_target
            mu: latent mean
            logvar: latent logvar
            x_target: shifted target tokens for loss computation
        """
        mu, logvar = self.encode(x)
        z = mu

        # --------------------
        # Teacher forcing (SHIFT)
        # --------------------
        x_input = x[:, :-1]    # input to decoder
        x_target = x[:, 1:]    # expected output

        logits = self.decode(z, x_input, teacher_forcing_ratio=0.95)

        return logits, mu, logvar, x_target

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
                probs = torch.softmax(logits, dim=-1)
                token = torch.argmax(probs, dim=-1)

                if token.item() == stoi["<eos>"]:
                    break

                output.append(token.item())
                x = token.unsqueeze(0)

            # Convert token IDs to string, removing special tokens
            return "".join([
                itos[i] for i in output
                if i not in (stoi["<sos>"], stoi["<pad>"])
            ])
