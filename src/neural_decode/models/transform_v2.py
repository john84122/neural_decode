from hflayers import Hopfield
from torch_brain.nn import FeedForward

import torch
import torch.nn as nn

from neural_decode.dataset.helper_functions import bin_spikes, generate_sinusoidal_position_embs

class TransformerHopfieldDecoder_v2(nn.Module):
    def __init__(
        self, num_units, bin_size, sequence_length,
        dim_output, dim_hidden, n_layers, n_heads,
    ):
        super().__init__()

        self.num_timesteps = int(sequence_length / bin_size)
        self.bin_size = bin_size

        self.readin = nn.Linear(num_units, dim_hidden)
        self.readout = nn.Linear(dim_hidden, dim_output)

        self.position_embeddings = nn.Parameter(
            data=generate_sinusoidal_position_embs(self.num_timesteps, dim_hidden),
            requires_grad=False,
        )

        self.transformer_layers = nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=dim_hidden,
                    num_heads=n_heads,
                    batch_first=True,
                ),
                Hopfield(
                    input_size=self.num_timesteps,
                    hidden_size=self.num_timesteps,
                    output_size=self.num_timesteps,
                    num_heads=n_heads,
                    batch_first=True,
                ),
                FeedForward(dim=dim_hidden),
            ])
            for _ in range(n_layers)
        ])

    def forward(self, x):
        x = self.readin(x)                              # (B, T, N) → (B, T, D)
        x = x + self.position_embeddings[None, ...]     # + position

        for attn, hopfield, ffn in self.transformer_layers:

            # Attention: temporal relationship (T)
            x_attn = attn(x, x, x, need_weights=False)[0]  # (B, T, D)

            # Hopfield: relationship between neurons (D)
            x_hop = hopfield(
                x.transpose(1, 2)                       # (B, T, D) → (B, D, T)
            ).transpose(1, 2)                           # (B, D, T) → (B, T, D)

            x = x + x_attn + x_hop                     # combine
            x = x + ffn(x)

        x = self.readout(x)                             # (B, T, D) → (B, T, 2)
        return x

    def tokenize(self, data):
        x = bin_spikes(
            spikes=data.spikes,
            num_units=len(data.units),
            bin_size=self.bin_size,
            num_bins=self.num_timesteps,
        ).T

        y = data.cursor.vel

        return {
            "model_inputs": {
                "x": torch.tensor(x, dtype=torch.float32),
            },
            "target_values": torch.tensor(y, dtype=torch.float32),
        }