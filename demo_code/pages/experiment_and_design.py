
import pandas as pd
import streamlit as st

import plotly.express as px

def main():
    st.write("# Experiment and Design")

    st.write("For our experiment, we design the pipelin as shown below:")
    st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/misc/document_design_img.png", caption="Image of our Initial Model Pipeline. Note that our setup does differ a bit from this. Specifically, we do not use captum for computing attribution maps. Second, we do not do a separate test accuracy. We only use a validation and train dataset.")

    st.write("Our experiments consist of 4 different components")
    st.write("1. We preprocess and choose a dataset (preprocessing and dataset partitioning).")
    st.write("2. We build three different model architectures (model selection).")
    st.write("3. We train and evaluate our models on a validation set (training and accuracy evaluation).")
    st.write("4. We perform embedding and attribution based analysis (post hoc analysis).")
    st.write()
    st.write("Each of these components are discussed in detail below.")

    st.divider()
    st.write("## Dataset")

    st.write("The dataset that we use is the Perich Miller Decoding Dataset that we show in the background.")
    st.write("Specifics on Train-Test Split, configurations, and others is shown below.")

    dataset_table = pd.DataFrame({
        "Train Batches" : [10],
        "Test Batches": [2],
        "Dimension of Input": [(100, 55)],
        "Temporal Window": [1],
        "Augmentations": ["Binning, Sinusoidal Positional Embeddings"],
    })

    st.table(dataset_table)

    st.write("Note that the inputs to the model are not the raw spikes, but rather binned spikes which count the amount of activity measured within a 1 second interval (temporal window).")

    st.divider()
    st.write("## Models")

    st.write("We use three different types of models for training, with the differences between each model shown below:")

    st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/misc/model_architectures.png")

    st.write("Each model contains five layers layers")
    st.write("- Each model contains both a read in and read out layer.")
    st.write("- Outside of this, each model contains 3 layers consistent of attention heads or hopfield layers.")
    st.write("- Attention layers have 4 heads")

    st.divider()
    st.write("## Training Configuration")

    st.write("For our Training Configuration, we use gradient descent on the training set. Basic configurations are shown in the table below:")

    train_table = pd.DataFrame({
        "learning rate" : [1e-3],
        "Num Epochs": [100],
        "Optimizer": ["AdamW"],
        "Loss": ["MSE"],
    })

    st.table(train_table)

    st.divider()
    st.write("## Post Hoc Analysis")
    st.write("Finally, we implement some basic post hoc analysis. This includes UMAP embeddeing analysis of the last layers of each model. This embedding analysis learns a lower dimensional representation of high dimensional data which attempts to preserve topology. Metric used for umap analysis is euclidean distance and 3 components learned. We qualitatively measure the similarity between all three models.")
    st.write("")
    st.write(" In addition, we produce attribution maps similar to Simonyan et al. These attribution maps are gradient of the squred error between predicted and true validation set inputs with the respect to the input layer. The idea is that the larger gradient, the more important the feature is. We compute distances between normalized attribution maps to find where each model is the most similar and the most different in how it makes decisions. A final piece of analysis is looking at the mean differences between attribution maps to see how models relate to each other in how they do this regression task.")
    
    st.divider()
    st.write("## Evaluation of Success")
    st.write("We use R2 correlation between the predicted and true validation velocities between the two models in order to determine whether these models were successful at learning the task. We expect that the model will perform well if each model scores an R2 validation above .5 becasue this is the score that indicates there is a strong linear correlation between scores in any statistics background.")
    st.write()
    st.write("We will also consider whether models are different from each other using statistical tests (1 sample t-test) on the differences between explanations on the attribution maps. The idea is that we hope that the differences between explanations has a mean value no different from zero (p value not significant). This should be some indication that explanations should not significantly different between any two models.")
    st.write("Else, the UMAP embeddings will be qualitatively analyzed and use to determine difference between their embedding spaces.")


if __name__ == "__main__":
    main()