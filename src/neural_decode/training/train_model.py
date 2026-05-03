'''
    Basic scripts to do 
'''

import torch.nn.functional as F
from neural_decode.evaluation.metrics_for_performance import compute_r2, move_to_gpu



def train(model, optimizer, train_loader, val_loader, num_epochs=50, store_embs=False, device = "cpu"):
    # We'll store some intermediate outputs for visualization
    train_outputs = {
        'n_epochs': num_epochs,
        'unit_emb': [],
        'session_emb': [],
        'output_pred': [],
        'output_gt': [],
    }

    r2_log = []
    loss_log = []

    # Training loop
    for epoch in range(num_epochs):
        # Compute R² score on validation set
        r2, target, pred = compute_r2(val_loader, model)
        r2_log.append(r2)

        # Training steps
        for batch in train_loader:
            batch = move_to_gpu(batch, device)
            loss = training_step(batch, model, optimizer)
            loss_log.append(loss.item())

        print(f"\rEpoch {epoch+1}/{num_epochs} | Val R2 = {r2:.3f} | Loss = {loss.item():.3f}", end="")

        # Store intermediate outputs
        if store_embs:
            train_outputs['unit_emb'].append(model.unit_emb.weight[1:].detach().cpu().numpy())
            train_outputs['session_emb'].append(model.session_emb.weight[1:].detach().cpu().numpy())
        train_outputs['output_gt'].append(target.detach().cpu().numpy())
        train_outputs['output_pred'].append(pred.detach().cpu().numpy())

    # Compute final R² score
    r2, _, _ = compute_r2(val_loader, model)
    r2_log.append(r2)
    print(f"\nDone! Final validation R2 = {r2:.3f}")

    return r2_log, loss_log, train_outputs


def training_step(batch, model, optimizer):
    optimizer.zero_grad()                  # Step 0. Clear old gradients
    pred = model(**batch["model_inputs"])  # Step 1. Do forward pass
    target = batch["target_values"]
    loss = F.mse_loss(pred, target)        # Step 2. Compute loss
    loss.backward()                        # Step 3. Backward pass
    optimizer.step()                       # Step 4. Update model params
    return loss

def main():
    pass

if __name__ == "__main__":
    main()