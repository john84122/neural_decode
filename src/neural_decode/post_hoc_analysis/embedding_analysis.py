'''
A simple script to study the embeddings of the model and see how to qualitatively evaluate them.
'''

import torch
import numpy as np

from umap import UMAP
import plotly.express as px

def collect_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    
    embeddings = {}
    def hook_for_readout(module, input, output):
        embeddings['last'] = output.detach()

    hook = model.readout.register_forward_hook(
        lambda mod, inp, out: embeddings.update({'last': inp[0].detach()})
    )

    with torch.no_grad():
        for batch in dataloader:

            labels = batch["target_values"]

            model(**batch["model_inputs"])
            
            all_embeddings.append(embeddings["last"].cpu())
            all_labels.append(labels.cpu())
    
    hook.remove()

    all_embeddings = torch.cat(all_embeddings, dim=0).detach().cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).detach().cpu().numpy()
    
    return all_embeddings, all_labels


def flatten_embeddings_and_labels(embeddings, labels):
    
    output_embeddings = []
    output_labels = []

    for k in range(embeddings.shape[0]):
        for t in range(embeddings.shape[1]):
            output_embeddings.append(embeddings[k, t, :])
            output_labels.append(np.linalg.norm(labels[k, t]))

    return np.array(output_embeddings), np.array(output_labels)

def compute_umap_embeddings(model, dataloader, n_components=3, metric="euclidean", device = "cpu"):

    embeddings, labels = collect_embeddings(model, dataloader, device=device)
    out, lbs_mag = flatten_embeddings_and_labels(embeddings, labels)

    embedder = UMAP(n_components=n_components, metric=metric)
    umap_embeddings = embedder.fit_transform(out)

    return umap_embeddings, lbs_mag

def main():
    pass

if __name__ == "__main__":
    main()