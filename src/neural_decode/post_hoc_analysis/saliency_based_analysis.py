
import numpy as np

def get_attribution(model, dataset):
    model.eval()
    attrib_output = []

    for batch in dataset:
        labels = batch["target_values"]

        x_inp = batch["model_inputs"]["x"].requires_grad_(True)

        out = model(x_inp)
        loss = ((out - labels) ** 2).sum()
        loss.backward()

        attrib_output.append(x_inp.grad.abs().detach().cpu().numpy())

    return attrib_output

def return_normalized_and_aggregated_attribution(attrib_output):

    stacked_attrib = np.concatenate(attrib_output, axis=0)

    normed_list = []

    for i in range(stacked_attrib.shape[0]):
        normaliized_saliency = (stacked_attrib[i] - np.min(stacked_attrib[i]))
        normaliized_saliency = normaliized_saliency / np.max(normaliized_saliency)
        normed_list.append(normaliized_saliency)

    normed_attrib = np.stack(normed_list, axis=0)

    return normed_attrib

def compute_and_return_attribution_maps(models, dataset):

    attribution_vals = get_attribution(models, dataset)

    normalized_attribution = return_normalized_and_aggregated_attribution(attribution_vals)

    return normalized_attribution

def main():
    pass

if __name__ == "__main__":
    main()