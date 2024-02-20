import torch
import torch.nn as nn

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

