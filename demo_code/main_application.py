import pandas as pd
import streamlit as st


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.write("## Neural Decoding via Transformer and Hopfield Models")
    
    c1, c2 = st.columns(2, vertical_alignment="center")
    with c1:
        st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/misc/icon_for_git_page.png", width=500)

    with c2:
        st.markdown(
            "<h2 style='text-align: center;'>Decoding the Illusive Spiking Activity of the Mind</h2>", 
            unsafe_allow_html=True
        )
        st.markdown(
            "<h6 style='text-align: center;'>Created by Elif Ercek and Johannes Bauer</h6>", 
            unsafe_allow_html=True
        )
        st.write("")

    st.divider()

    st.write("# Motivation")
    st.write('''Neural decoding is the task of interpreting spiking activity in a animal brain and mapping it to actions of the animal. Given the complex dynamics of the brain, this is a nontrivial task. To solve this problem, deep neural networks, such as transformers have been useful for neural decoding tasks. However, we seek to understand how a variety of models, including Hopfield Models, Hybrid Models, and Transformers perform on Neural Decoding datasets.''')
    st.write('''  ''')
    st.write('''In addition, a second motivation of this experiment is that some work has suggested one can produce hopfield networks that are equivalent in nature to transformers. In this paper (Ramsauer et al.), they provide some evidence that this is true, however they only use limited number of datasets to show this equivalence. Thus, there is a need for quantitative, rather than merely theoretical, evidence to show equivalence between these two models. In addition, there are questions around whether hybrid Hopfield and Transformer models can be equivalent to both Hopfield Networks and Transformer Models. Both concepts are what we attempt to study in our experiments.''')

    st.divider()

    st.write("# Main Research Questions")

    st.write("The Main research questions that we want to answer are:")
    st.write("  - How equivalent are Transformer Models to Hopfield Networks in their performance on Neural Decoding Tasks?")
    st.write("  - Are transformers and Hopfield Networks good for Neural Decoding?")

    st.divider()

    st.write("## Navigation")
    st.write("In this work, we have five different sections. Each are listed and described below.")

    clmns = st.columns(5)

    clmns_out = st.columns(2, vertical_alignment="center")

    if "active_section" not in st.session_state:
        st.session_state.active_section = None

    with clmns[0]:
        if st.button("Background"):
            st.session_state.active_section = "background"

    with clmns[1]:
        if st.button("Experiments"):
            st.session_state.active_section = "experiments"

    with clmns[2]:
        if st.button("Results"):
            st.session_state.active_section = "results"

    with clmns[3]:
        if st.button("Explainability"):
            st.session_state.active_section = "xai"

    with clmns[4]:
        if st.button("Summary"):
            st.session_state.active_section = "summary"

    if st.session_state.active_section == "background":

        with clmns_out[0]:
            st.write("## Background")
            st.write("In this section, we discuss the main ideas and motivations of the project. This includes:")
            st.write('''
                    - Neural Decoding Tasks
                    - Transformers
                    - Hopfield Networks
                    - Equivalences between Transformers and Hopfield Networks
                     ''')

        if st.button("Go To Page", key = "background"):
            st.switch_page("pages/background.py")


        with clmns_out[1]:
            st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/real_images/image.png", caption="Image from Perich et al., 2018", width="stretch")
            # https://www.cell.com/neuron/fulltext/S0896-6273(18)30832-8?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0896627318308328%3Fshowall%3Dtrue


    if st.session_state.active_section == "experiments":

        with clmns_out[0]:
            st.write("## Experiments")
            st.write("In this section, we discuss our experiments. This section focuses on:")
            st.write('''
                    - Our Main Problem
                    - Datasets
                    - Models
                    - Train Setup
                    - Evaluation and Post Analysis Setup
                     ''')

        if st.button("Go To Page", key="experiments"):
            st.switch_page("pages/experiments_and_design.py")


        with clmns_out[1]:
            st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/misc/hopfield_layer.png", width="stretch", caption="Hopfield Layer from Ramsauer et al, 2021")

    if st.session_state.active_section == "results":

        with clmns_out[0]:
            st.write("## Summary")
            st.write("For this section, we present the training results and outcomes. This includes:")
            st.write('''
                    - Validation Scores
                    - Performance Trends
                    - Key Conclusions of Our Analysis
                     ''')

        if st.button("Go To Page", key="summary"):
            st.switch_page("pages/training_performance.py")


        with clmns_out[1]:
            st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/real_images/results_for_model.png", width="stretch")

    if st.session_state.active_section == "xai":

        with clmns_out[0]:
            st.write("## Embeddings and Attribution")
            st.write("We will present embeddings and saliency maps analysis")
            st.write('''
                    - UMAP Embeddings for Each Model
                    - Saliency Maps of the Models
                    - Conclusiosn From the Qualitative Analysis
                     ''')

        if st.button("Go To Page", key="xai"):
            st.switch_page("pages/explainability.py")


        with clmns_out[1]:
            st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/misc/umap_image.png", width="stretch", caption = "Image is from McInnes et al., 2018")

    if st.session_state.active_section == "summary":

        with clmns_out[0]:
            st.write("## Summary ")
            st.write("Finally, this section summarizes the main findings. This includes:")
            st.write('''
                    - Discussion on Similarity Between Models
                    - Limitations of Work
                    - Future Directions
                     ''')
        #go_to_bt = st.button("Go To Page")

        if st.button("Go To Page", key="summary"):
            st.switch_page("pages/conclusions.py")


        with clmns_out[1]:
            st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/misc/summary.jpeg", width="stretch")
