import pandas as pd
import streamlit as st


if __name__ == "__main__":

    st.write("#Post Hoc Analysis of Transformer Models")


    st.write("## Graph of UMAP Embeddings for Each Model")
    plot_options = {"Transformer Embeddings", "Hopfield Embeddings"}

    col1, col2 = st.columns(2)

    with col1:
        show_transform = st.button("Transformer Embeddings")

    with col2:
        show_hopfield = st.button("Hopfield Embeddings")

    if show_transform == True:
            st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/umap_img_fanfic_1.png")

    if show_hopfield == True:
        st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/umap_img_fanfic_2.png")

st.divider()
st.write("## Saliency Analysis")
cols_for_saliency = st.columns(3)

with cols_for_saliency[0]:
     st.write("Similarity: 5")
     st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/importance_1.png")
     st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/importance_2.png")
     bt_1 = st.button("Show Saliency 1")


with cols_for_saliency[1]:
     st.write("Similarity: 15")
     st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/importance_1.png")
     st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/importance_2.png")
     bt_2 = st.button("Show Saliency 2")

with cols_for_saliency[2]:
     st.write("Similarity: 45")
     st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/importance_1.png")
     st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/importance_2.png")
     bt_3 = st.button("Show Saliency 3")

if bt_1 == True:
     st.write("### Close up Visualization of Saliency Outputs")
     st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/importance_1.png")
     st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/importance_2.png")