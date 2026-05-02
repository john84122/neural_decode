
from torch_brain.utils import seed_everything

from neural_decode.models.hopfield_only import HopfieldOnlyDecoder


def test_hopfield_only_training():
    seed_everything(0)

    # 1. Setup datasets and dataloader
    recording_id = "/Users/johannesbauer/Documents/Coding/neuro_comp_project/data/perich_miller_population_2018/t_20130819_center_out_reaching"
    train_dataset, train_loader, val_dataset, val_loader = get_train_val_loaders(recording_id, batch_size=64)
    num_units = len(train_dataset.get_unit_ids())
    print(f"Num Units in Session: {num_units}")

# 2. Initialize Model with the new MLP definition
    mlp_model = MLPNeuralDecoder(
        num_units=num_units,    # Num. of units inputted (spiking activity)
        #
        bin_size=10e-3,         # Duration (s) of bins
        sequence_length=1.0,    # Context length of the model
        #
        output_dim=2,           # Output dimension of final readout layer
        hidden_dim=32,          # Hidden dimension of the model
    )
    mlp_model = mlp_model.to(device)

    # 3. Connect Tokenizer to Datasets
    transform = mlp_model.tokenize
    train_dataset.transform = transform
    val_dataset.transform = transform

    # 4. Setup Optimizer
    optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=1e-3)

    # 5. Train!
    mlp_r2_log, mlp_loss_log, mlp_train_outputs = train(mlp_model, optimizer, train_loader, val_loader, num_epochs=100)

    # Plot the training loss and validation R2
    plot_training_curves(mlp_r2_log, mlp_loss_log)




def main():
    pass

if __name__ == "__main__":
    main()