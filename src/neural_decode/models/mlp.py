import torch
import torch.nn as nn

from neural_decode.dataset.helper_functions import bin_spikes

class MLPNeuralDecoder(nn.Module):
    def __init__(self, num_units, bin_size, sequence_length, output_dim, hidden_dim):
        """Initialize the neural net layers."""
        super().__init__()

        self.num_timesteps = int(sequence_length / bin_size)
        self.bin_size = bin_size

        self.net = nn.Sequential(
            nn.Linear(self.num_timesteps * num_units, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * self.num_timesteps),
        )

    def forward(self, x):
        """Produces predictions from a binned spiketrain.
        This is pure PyTorch code.

        Shape of x: (B, T, N)
        """

        x = x.flatten(1)                          # (B, T, N)    -> (B, T*N)
        x = self.net(x)                           # (B, T*N)     -> (B, T*D_out)
        x = x.reshape(-1, self.num_timesteps, 2)  # (B, T*D_out) -> (B, T, D_out)
        return x

    def tokenize(self, data):
        """tokenizes a data sample, which is a sliced Data object"""

        # A. Extract and bin neural activity (data.spikes)
        spikes = data.spikes
        x = bin_spikes(
            spikes=spikes,
            num_units=len(data.units),
            bin_size=self.bin_size,
            num_bins=self.num_timesteps
        ).T
        # Final shape of x here is (timestamps, num_neurons)

        # B. Extract the corresponding cursor velocity, which will act as targets
        #    for training the MLP.
        y = data.cursor.vel
        # Final shape of y is (timestamps x 2)
        # Note that in this example we have choosen the bin size to match the
        # sampling rate of the recorded cursor velocity.

        # Finally, we output the "tokenized" data in the form of a dictionary.
        data_dict = {
            "model_inputs": {
                "x": torch.tensor(x, dtype=torch.float32),
                # Models in torch_brain typically follow the convention that
                # fields that are input to model.forward() are stored in
                # "model_inputs". Although you are free to deviate from this,
                # we have found that this convention generally produces cleaner
                # training loops.
            },
            "target_values": torch.tensor(y, dtype=torch.float32),
        }
        return data_dict

def main():
    pass

if __name__ == "__main__":
    main()