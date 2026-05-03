import os
import torch

import numpy as np
import matplotlib.pyplot as plt

from torch_brain.utils import seed_everything
from neural_decode.dataset.transformer_dataloader import get_train_val_loaders

from neural_decode.models.transformer import TransformerNeuralDecoder 
from neural_decode.models.hopfield_only import HopfieldOnlyDecoder
from neural_decode.models.transformer_hopfield import TransformerHopfieldDecoder

from neural_decode.training.train_model import train
from neural_decode.evaluation.graphing_functions import plot_training_curves


from neural_decode.post_hoc_analysis.embedding_analysis import compute_umap_embeddings
from neural_decode.post_hoc_analysis.saliency_based_analysis import compute_and_return_attribution_maps

seed_everything(0)


print("--------Setting Up of Analysis--------")
print()
base_dir = os.getcwd().split("additional")[0]
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using: {device}")
print()

print("--------Loading Dataset Outputs--------")
path_to_neural_dataset = os.path.join(base_dir, "data", "perich_miller_population_2018", "t_20130819_center_out_reaching")
train_dataset, train_loader, val_dataset, val_loader = get_train_val_loaders(path_to_neural_dataset, batch_size=64)

num_units = len(train_dataset.get_unit_ids())
print()
print(f"Batches in Train Dataset: {len(train_loader)}")
print(f"Batches in Val Dataset: {len(val_loader)}")
print(f"Num Units in Session: {num_units}")
print()

print()

print("--------Setting Up of Models--------")
print()
tf_model = TransformerNeuralDecoder(
    num_units=num_units,
    bin_size=10e-3,
    sequence_length=1.0,
    dim_output=2,
    dim_hidden=128,
    n_layers=3,
    n_heads=4,
).to(device)

hopfield_only_model = HopfieldOnlyDecoder(
    num_units=num_units,
    bin_size=10e-3,
    sequence_length=1.0,
    dim_output=2,
    dim_hidden=128,
    n_layers=3,
).to(device)

tf_hopfield_model = TransformerHopfieldDecoder(
    num_units=num_units,
    bin_size=10e-3,
    sequence_length=1.0,
    dim_output=2,
    dim_hidden=128,
    n_layers=3,
    n_heads=4,
).to(device)

print("*****Transformer Neural Decoder*****")
print(tf_model)
print()
print("*****Hopfield Only Decoder*****")
print(hopfield_only_model)
print()
print("*****Transformer + Hopfield Decoder*****")
print(tf_hopfield_model)
print()

print("--------Training Models--------")
print()
train_dataset.transform = tf_model.tokenize
val_dataset.transform = tf_model.tokenize

optimizer = torch.optim.AdamW(tf_model.parameters(), lr=1e-3)

print("*****Transformer Neural Decoder*****")
transformer_r2_log_tf, transformer_loss_log_tf, transformer_train_outputs_tf = train(tf_model, optimizer, train_loader, val_loader,
                                                                            num_epochs=3, device = device)

print()

print("*****Hopfield Only Decoder*****")
transformer_r2_log_hop, transformer_loss_log_hop, transformer_train_outputs_hop = train(hopfield_only_model, optimizer, train_loader, val_loader,
                                                                            num_epochs=3, device = device)

print()

print("*****Transformer + Hopfield Decoder*****")
transformer_r2_log_tf_hop, transformer_loss_log_tf_hop, transformer_train_outputs_tf_hop = train(tf_hopfield_model, optimizer, train_loader, val_loader,
                                                                            num_epochs=3, device = device)

print()

print("--------Evaluating Models--------")
print()
fig_plot_tf = plot_training_curves(transformer_r2_log_tf, "R2 Scores for Transformer Decoder")

fig_plot_hop = plot_training_curves(transformer_r2_log_hop, "R2 Scores for Hopfield Only Decoder")

fig_plot_tf_hop = plot_training_curves(transformer_r2_log_tf_hop, "R2 Scores for Transformer + Hopfield Decoder")

fig_plot_tf.savefig(os.path.join(base_dir, "results", "quick_run_results", "training_curves_tf.png"))
fig_plot_hop.savefig(os.path.join(base_dir, "results", "quick_run_results", "training_curves_hop.png"))
fig_plot_tf_hop.savefig(os.path.join(base_dir, "results", "quick_run_results", "training_curves_tf_hop.png"))
print("Finished saving training curves - see the results folder.")
print()

print("--------Embedding Analysis--------")
print()
embeddings, computations = compute_umap_embeddings(tf_hopfield_model, val_loader, device=device)
print(f"UMAP Embeddings Shape: {embeddings.shape}")

print()
print("--------Saliency Map Analysis--------")
print()
sal_maps = compute_and_return_attribution_maps(tf_hopfield_model, val_loader)
print(f"Saliency Maps Shape: {sal_maps.shape}")
print(f"Min Value in Saliency Maps: {np.min(sal_maps)}")
print(f"Max Value in Saliency Maps: {np.max(sal_maps)}")

print()
print("DONE!")
print()