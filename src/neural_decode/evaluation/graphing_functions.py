'''
Simple graphing functions for plotting training curves and other relevant metrics for the trained models.
'''

import numpy as np
import matplotlib.pyplot as plt

def plot_training_curves(relevant_score, title):
    '''
    Plots the training curves for a arbitrary relevant_score. The relevant scores should be some array like object and you can define the title which is a string.
    '''
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(relevant_score, label = "r2 Scores")
    ax.set_xlabel("Epoch", weight = "bold", size = 13)
    ax.set_ylabel("R² Score", weight = "bold", size = 13)
    ax.set_title(title, weight = "bold", size = 15)

    return fig


def plot_training_curves_wholistic(r2_log, loss_log):
    """
    Plots the training curves: training loss and validation R2 score.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(np.linspace(0, len(loss_log), len(loss_log)), loss_log)
    plt.title("Training Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("MSE Loss")
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(r2_log)
    plt.title("Validation R2")
    plt.xlabel("Epochs")
    plt.ylabel("R2 Score")
    plt.grid()
    plt.tight_layout()
    plt.show()

def main():
    pass

if __name__ == "__main__":
    main()