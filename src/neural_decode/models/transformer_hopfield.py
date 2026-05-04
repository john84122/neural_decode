'''
Defines the transformer + hopfield model used in experiments.
'''
import torch
import torch.nn as nn

from hflayers import Hopfield
from torch_brain.nn import FeedForward

from neural_decode.dataset.helper_functions import bin_spikes, generate_sinusoidal_position_embs

class TransformerHopfieldDecoder(nn.Module):
    def __init__(
        self, num_units, bin_size, sequence_length,
        dim_output, dim_hidden, n_layers, n_heads,
    ):
        '''
        Defines the transformer + hopfield model. This model contains of multiple attention heads and one second to last hopfield layer.
        '''
        super().__init__()

        self.num_timesteps = int(sequence_length / bin_size)
        self.bin_size = bin_size

        # Read-in / Read-out
        self.readin = nn.Linear(num_units, dim_hidden)
        self.readout = nn.Linear(dim_hidden, dim_output)

        # Positional embedding
        self.position_embeddings = nn.Parameter(
            data=generate_sinusoidal_position_embs(self.num_timesteps, dim_hidden),
            requires_grad=False,
        )

        # Transformer layers
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

        #  Hopfield layer (Only one)
        self.hopfield = Hopfield(
            input_size=dim_hidden,
            hidden_size=dim_hidden,
            output_size=dim_hidden,
            num_heads=4,
            batch_first=True
        )

    def forward(self, x):
        '''
        The forward pass of the model.
        '''
        # Read-in
        x = self.readin(x)

        # Positional encoding
        x = x + self.position_embeddings[None, ...]

        # Transformer + Hopfield
        for i, (attn, ffn) in enumerate(self.transformer_layers):

            x = x + attn(x, x, x, need_weights=False)[0]
            if i == len(self.transformer_layers) - 1: #only last layer
              x = x + self.hopfield(x)

            x = x + ffn(x)

        # Read-out
        x = self.readout(x)

        return x

    def tokenize(self, data):
        '''
        Method for tokenizing the data before training. Call using model.transform = model.tokenize
        '''


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