import torch
import torch.nn as nn
from torch_brain.nn import FeedForward

from neural_decode.dataset.helper_functions import bin_spikes, generate_sinusoidal_position_embs

class TransformerNeuralDecoder(nn.Module):
    def __init__(
        self, num_units, bin_size, sequence_length,   # data properties
        dim_output, dim_hidden, n_layers, n_heads,    # transformer properties
    ):
        """Initialize the neural net components"""
        super().__init__()

        self.num_timesteps = int(sequence_length / bin_size)
        self.bin_size = bin_size

        # Create the read-in/out linear layers
        self.readin = nn.Linear(num_units, dim_hidden)
        self.readout = nn.Linear(dim_hidden, dim_output)

        # Create the position embeddings
        # Note that these are kept constant in this implementation, i.e. _not_ learnable
        self.position_embeddings = nn.Parameter(
            data=generate_sinusoidal_position_embs(self.num_timesteps, dim_hidden),
            requires_grad=False,
        )

        # Create the transformer layers:
        # each composed of the Attention and the feedforward (FFN) blocks
        self.transformer_layers = nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=dim_hidden,
                    num_heads=n_heads,
                    batch_first=True,
                ),
                FeedForward(dim=dim_hidden),
            ])
            for _ in range(n_layers)
        ])

    def forward(self, x):
        """Produces predictions from a binned spiketrain.
        This is pure PyTorch code.

        Shape of x: (B, T, N)
        """

        # Read-in: converts our input marix to transformer tokens; one token for each timestep
        x = self.readin(x)  # (B, T, N) -> (B, T, D)

        # Add position embeddings to the tokens
        x = x + self.position_embeddings[None, ...]  # -> (B, T, D)

        # Transformer
        for attn, ffn in self.transformer_layers:
            x = x + attn(x, x, x, need_weights=False)[0]
            x = x + ffn(x)

        # Readout: converts tokens to 2d vectors; each vector signifying (v_x, v_y) at that timestep
        x = self.readout(x)  # (B, T, D) -> (B, T, 2)

        return x

    def tokenize(self, data):
        # Same tokenizer as the MLP

        # A. Bin spikes
        x = bin_spikes(
            spikes=data.spikes,
            num_units=len(data.units),
            bin_size=self.bin_size,
            num_bins=self.num_timesteps,
        ).T

        # B. Extract targets
        y = data.cursor.vel

        data_dict = {
            "model_inputs": {
                "x": torch.tensor(x, dtype=torch.float32),
            },
            "target_values": torch.tensor(y, dtype=torch.float32),
        }
        return data_dict