'''
    Simple script to evaluate the trained models.
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import warnings
import logging
from torch_brain.utils import seed_everything

def move_to_gpu(data, device):
    """
    Recursively moves tensors (or collections of tensors) to the given device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_gpu(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_gpu(elem, device) for elem in data]
    else:
        return data


def r2_score(y_pred, y_true):
    '''
    Computes the R^2 scores for a given collection of predictions and true values. This is mainly used for the validation set.
    '''
    # Compute total sum of squares (variance of the true values)
    y_true_mean = torch.mean(y_true, dim=0, keepdim=True)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)

    # Compute residual sum of squares
    ss_res = torch.sum((y_true - y_pred) ** 2)

    # Compute R^2
    r2 = 1 - ss_res / ss_total

    return r2

def compute_r2(dataloader, model, device = "cpu"):
    '''
    Computes the R^2 scores given a dataloader, model, and device. Note that this is the main function which should be use for evlauating the models, not the r2_score function directly.
    '''
    # Compute R2 score over the entire dataset
    total_target = []
    total_pred = []
    for batch in dataloader:
        batch = move_to_gpu(batch, device)
        pred = model(**batch["model_inputs"])
        target = batch["target_values"]

        # Store target and pred for visualization
        mask = torch.ones_like(target, dtype=torch.bool)
        if "output_mask" in batch["model_inputs"]:
            mask = batch["model_inputs"]["output_mask"]
        total_target.append(target[mask])
        total_pred.append(pred[mask])

    # Concatenate all batch outputs
    total_target = torch.cat(total_target)
    total_pred = torch.cat(total_pred)

    # Compute the R2 score
    r2 = r2_score(total_pred.flatten(), total_target.flatten())

    return r2.item(), total_target, total_pred

def print_model(model: torch.nn.Module):
    """
    Prints a summary of the model architecture and parameter count.
    """
    model_str = str(model).split('\n')
    print("\nModel:")
    print('\n'.join(model_str[:5]))
    print("...")
    print('\n'.join(model_str[-min(5, len(model_str)):]))
    num_params = sum(p.numel() for p in model.parameters())
    if num_params > 1e9:
        param_str = f"{num_params/1e9:.1f}G"
    elif num_params > 1e6:
        param_str = f"{num_params/1e6:.1f}M"
    else:
        param_str = f"{num_params/1e3:.1f}K"
    print(f"\nNumber of parameters: {param_str}\n")
