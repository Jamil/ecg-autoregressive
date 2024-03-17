import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerBlock(nn.Module):
    def __init__(self, latent_dim, num_heads, dim_feedforward):
        super(TransformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_heads)
        self.layer_norm2 = nn.LayerNorm(latent_dim)
        self.transformer_dense1 = nn.Linear(latent_dim, dim_feedforward)
        self.transformer_dense2 = nn.Linear(dim_feedforward, latent_dim)

    def forward(self, x):
        x1 = self.layer_norm1(x)
        x1 = x1.permute(1, 0, 2)  # Adjust for batch_first in MultiheadAttention
        attention_output, _ = self.attn(x1, x1, x1)
        attention_output = attention_output.permute(1, 0, 2)  # Revert permutation
        x2 = attention_output + x
        x3 = self.layer_norm2(x2)
        x3 = torch.relu(self.transformer_dense1(x3))
        x3 = torch.relu(self.transformer_dense2(x3))
        x = x3 + x2
        return x

class CombinedModel(nn.Module):
    def __init__(self, input_dim, latent_dim, num_transformer_blocks=1, encoder_hidden_dim=64, dim_feedforward=256, num_heads=4):
        super(CombinedModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, input_dim)
        )
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(latent_dim, num_heads=num_heads, dim_feedforward=dim_feedforward)
            for _ in range(num_transformer_blocks)
        ])
        self.transformer_flatten = nn.Flatten()
        self.transformer_decoder1 = nn.Linear(10 * latent_dim, dim_feedforward)
        self.transformer_decoder2 = nn.Linear(dim_feedforward, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.transformer_flatten(x)
        x = torch.relu(self.transformer_decoder1(x))
        x = self.transformer_decoder2(x)
        x = self.decoder(x)
        return x

class MLP(nn.Module):
    # A simple 2-layer MLP for encoding and decoding
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class TEModel(nn.Module):
    def __init__(self, input_dim, latent_dim, num_transformer_blocks=1, encoder_hidden_dim=64, dim_feedforward=256, num_heads=4):
        super(TEModel, self).__init__()
        # Encoder MLP: Maps from raw patch_size to latent_dim
        self.encoder_mlp = MLP(input_dim, latent_dim, encoder_hidden_dim)

        # Transformer encoder setup
        encoder_layers = TransformerEncoderLayer(latent_dim, num_heads, dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_transformer_blocks)

        # Decoder MLP: Maps from latent_dim back to input_dim
        self.decoder_mlp = MLP(latent_dim, input_dim, encoder_hidden_dim)

        self.positional_encoding = nn.Parameter(torch.zeros(1, 10, latent_dim))  # Assuming seq_length=10

    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        batch_size, seq_length, _ = x.shape
        x = x.view(batch_size * seq_length, -1)
        x = self.encoder_mlp(x)  # Encode each patch
        x = x.view(batch_size, seq_length, -1)

        x += self.positional_encoding[:, :seq_length, :]  # Add positional encoding

        x = x.permute(1, 0, 2)  # Transformer expects (seq_length, batch_size, feature)
        transformer_output = self.transformer_encoder(x)
        transformer_output = transformer_output.permute(1, 0, 2)

        # Decode the last patch output
        last_patch_output = transformer_output[:, -1, :]
        predicted_patch = self.decoder_mlp(last_patch_output)
        predicted_patch = predicted_patch.view(batch_size, -1)  # Reshape to (batch_size, input_dim)

        return predicted_patch
