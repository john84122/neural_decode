import numpy as np
import matplotlib.pyplot as plt

def plot_training_curves(relevant_score):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(relevant_score, label = "r2 Scores")
    ax.set_xlabel("Epoch", weight = "bold", size = 13)
    ax.set_ylabel("R² Score", weight = "bold", size = 13)
    ax.set_title("Validation R Score over Training", weight = "bold", size = 15)

    return fig

def main():
    pass

if __name__ == "__main__":
    main()