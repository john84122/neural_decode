'''
A set of scripts that mainly help with loading the dataset for neural decoding tasks.
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import warnings
import logging
from torch_brain.utils import seed_everything

def generate_sinusoidal_position_embs(num_timesteps, dim):
    '''
    A augmentation function done to the positional embedding layers of each of the three models. Note: this is taken from
    the cosyne jupyter notebook on transformers. To find it, follow this link:
        - https://cosyne-tutorial-2025.github.io
    '''
    position = torch.arange(num_timesteps).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-np.log(10000.0) / dim))
    pe = torch.empty(num_timesteps, dim)
    pe[:, 0:dim // 2] = torch.sin(position * div_term)
    pe[:, dim//2:] = torch.cos(position * div_term)
    return pe

def bin_spikes(spikes, num_units, bin_size, right=True, num_bins=None):
    """
    Bins spike timestamps into a 2D array: [num_units x num_bins].
    """
    rate = 1 / bin_size  # avoid precision issues
    binned_spikes = np.zeros((num_units, num_bins))
    bin_index = np.floor((spikes.timestamps) * rate).astype(int)
    np.add.at(binned_spikes, (spikes.unit_index, bin_index), 1)
    return binned_spikes

def load_pretrained(ckpt_path, model):
    '''
    Loads the pretrained models from a given checkpoint path and model.
    '''
    print("Loading pretrained model...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # poyo is pretrained using lightning, so model weights are prefixed with "model."
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    print("Done!")
    return model


def get_dataset_config(brainset, sessions):
    '''
    Gets a configuration for the brain dataset.
    '''
    brainset_norms = {
        "perich_miller_population_2018": {
            "mean": 0.0,
            "std": 20.0
        }
    }

    config = f"""
    - selection:
      - brainset: {brainset}
        sessions:"""
    if type(sessions) is not list:
        sessions = [sessions]
    for session in sessions:
        config += f"""
          - {session}"""
    config += f"""
      config:
        readout:
          readout_id: cursor_velocity_2d
          normalize_mean: {brainset_norms[brainset]["mean"]}
          normalize_std: {brainset_norms[brainset]["std"]}
          metrics:
            - metric:
                _target_: torchmetrics.R2Score
    """

    config = OmegaConf.create(config)

    return config

def main():
    pass

if __name__ == "__main__":
    main()