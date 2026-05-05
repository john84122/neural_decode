import numpy as np
import pandas as pd
import streamlit as st


import plotly.express as px

if __name__ == "__main__":

     st.write("#Post Hoc Analysis of Transformer Models")

     st.write("In this section,we plotted and visualized the UMAP embeddings for each model. Note that each point represents a embedding of binned sensor outputs at a given time step. The total number of embeddings is 7300. Click on the buttons to see their embeddings.")

     st.write("## Graph of UMAP Embeddings for Each Model")
     plot_options = {"Transformer Embeddings", "Hopfield Embeddings"}

     col1, col2, col3 = st.columns(3)

     with col1:
          show_transform = st.button("Transformer")

     with col2:
          show_hopfield = st.button("Hopfield Only")

     with col3:
          show_tf_hop = st.button("Transformer + Hopfield")

     if show_transform == True:
          transform_embeddings = np.load("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/embeddings/tf/transformer_umap_embeddings.npy")
          transform_labels = np.load("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/embeddings/tf/transformer_umap_labels.npy")

          fig = px.scatter_3d(x=transform_embeddings[:,0], y=transform_embeddings[:,1], z=transform_embeddings[:,2], color=transform_labels,
                              color_continuous_scale=px.colors.sequential.Viridis)
          fig.update_layout(width=1000, height=500)
          st.plotly_chart(fig)


     if show_hopfield == True:
          hopfield_only_embeddings = np.load("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/embeddings/hopfield_only/hopfield_umap_embeddings.npy")
          hopfield_only_labels = np.load("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/embeddings/hopfield_only/hopfield_umap_labels.npy")

          fig = px.scatter_3d(x=hopfield_only_embeddings[:,0], y=hopfield_only_embeddings[:,1], z=hopfield_only_embeddings[:,2], color=hopfield_only_labels,
                              color_continuous_scale=px.colors.sequential.Magma)
          
          fig.update_layout(width=1000, height=500)
          st.plotly_chart(fig)

     if show_tf_hop == True:
          tf_hop_embeddings = np.load("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/embeddings/tf_hopfield/tf_hopfield_umap_embeddings.npy")
          tf_hop_labels = np.load("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/embeddings/tf_hopfield/tf_hopfield_umap_labels.npy")

          fig = px.scatter_3d(x=tf_hop_embeddings[:,0], y=tf_hop_embeddings[:,1], z=tf_hop_embeddings[:,2], color=tf_hop_labels,
                              color_continuous_scale=px.colors.sequential.Turbo)
          fig.update_layout(width=1000, height=500)
          st.plotly_chart(fig)

     st.write("Note that the transformer embeddings are most similar to the transformer + hopfield embeddings. Specifically, they have this very circular arc of points in which the magnitdue of the velocity is near zero. Outside of the arc is every other high velocity input. In contrast, the hopfield only embeddings are more snakelike and spread out. There is no indication that the arc that is clearly seen in the other embeddings is located. This indicates that there are differences between the embeddings spaces.")

     st.divider()
     st.write("## Saliency Analysis")
     st.write("Now, we conducted saliency map analysis. First, we computed attribution maps for instances in the validation set. Then, we computed the mean diffences between normalized attribution maps for each model on the same validation instance. Afterwards, we record the mean difference between cross comparison of attribution maps. In the plots below, we visualize the attribution maps that are most similar between all models, average simiarlity between the models, and the least similar between the models.")
     st.write()
     st.write("To look at each of these closer, please click the button associated to these attribution maps.")
     cols_for_saliency = st.columns(3)

     with cols_for_saliency[0]:
          st.write("Mean Similarity: 108.65")
          st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/saliency_map_imgs/sal_hop_low.png")
          st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/saliency_map_imgs/sal_hop_tf_low.png")
          st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/saliency_map_imgs/sal_tf_low.png")
          bt_1 = st.button("Show Saliency 1")


     with cols_for_saliency[1]:
          st.write("Similarity: 313.42")
          st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/saliency_map_imgs/sal_hop_med.png")
          st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/saliency_map_imgs/sal_hop_tf_med.png")
          st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/saliency_map_imgs/sal_tf_med.png")
          bt_2 = st.button("Show Saliency 2")

     with cols_for_saliency[2]:
          st.write("Similarity: 581.37")
          st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/saliency_map_imgs/sal_hop_high.png")
          st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/saliency_map_imgs/sal_hop_tf_high.png")
          st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/saliency_map_imgs/sal_tf_high.png")
          bt_3 = st.button("Show Saliency 3")

     if bt_1 == True:
          st.write("### Close up Visualization of Saliency Outputs that have high similarity")
          st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/saliency_map_imgs/sal_hop_low.png")
          st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/saliency_map_imgs/sal_hop_tf_low.png")
          st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/saliency_map_imgs/sal_tf_low.png")

     if bt_2 == True:
          st.write("### Close up Visualization of Saliency Outputs that are average similarity")
          st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/saliency_map_imgs/sal_hop_med.png")
          st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/saliency_map_imgs/sal_hop_tf_med.png")
          st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/saliency_map_imgs/sal_tf_med.png")

     if bt_3 == True:
          st.write("### Close up Visualization of Saliency Outputs that are low similarity")
          st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/saliency_map_imgs/sal_hop_high.png")
          st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/saliency_map_imgs/sal_hop_tf_high.png")
          st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/demo_images/saliency_map_imgs/sal_tf_high.png")

     st.write("What we find is that there are cases in which models decisions are very similar and also very different.")
     st.write("To caputre this difference, we can also plot the averge diffence between all models in a heatmap shown below")

     st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/misc/average_differences.png")
     st.write("Note that most models are actually quite similar to each other where mean difference comparisons are consistently the value of 3. But, note that the models that are the most different are the hopfield and the transformer from each other. This is incomparosn of any of these two more pure models and their differences from the hybrid transformer and hopfield network.")