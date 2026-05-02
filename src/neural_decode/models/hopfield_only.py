from hflayers import Hopfield
from torch_brain.nn import FeedForward

import torch
import torch.nn as nn

from neural_decode.dataset.helper_functions import bin_spikes, generate_sinusoidal_position_embs

class HopfieldOnlyDecoder(nn.Module):
    def __init__(
        self, num_units, bin_size, sequence_length,
        dim_output, dim_hidden, n_layers
    ):
        super().__init__()

        self.num_timesteps = int(sequence_length / bin_size)
        self.bin_size = bin_size

        # Read-in / Read-out
        self.readin = nn.Linear(num_units, dim_hidden)
        self.readout = nn.Linear(dim_hidden, dim_output)

        # Position embeddings
        self.position_embeddings = nn.Parameter(
            generate_sinusoidal_position_embs(self.num_timesteps, dim_hidden),
            requires_grad=False
        )

        # Hopfield layers
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Hopfield(
                    input_size=dim_hidden,
                    hidden_size=dim_hidden,
                    output_size=dim_hidden,
                    num_heads=4,
                    batch_first=True
                ),
                FeedForward(dim=dim_hidden)
            ])
            for _ in range(n_layers)
        ])

    def forward(self, x):
        # (B, T, N) → (B, T, D)
        x = self.readin(x)

        # + position
        x = x + self.position_embeddings[None, ...]

        # Hopfield blocks
        for hopfield, ffn in self.layers:
            x = x + hopfield(x)   # hopfield instead of attention
            x = x + ffn(x)

        # output
        x = self.readout(x)
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


def main():
    pass

if __name__ == "__main__":
    main()