import pandas as pd
import streamlit as st


if __name__ == "__main__":
    st.write("# Transforms versus Hopfield Networks for Neural Decoding Tasks")
    st.write("# Motivation and problems")
    st.write('''The goal of this project is to investigate whether Hopfield layers can replace multi-head self-attention in Transformer models for neural decoding tasks. The main research question is: Can Hopfield layers replace multi-head self-attention in Transformers while maintaining or improving performance and interpretability in neural decoding? To answer this, we will design a Transformer model and a corresponding Hopfield-based model that are theoretically equivalent. We will train both models on neural decoding data to predict reaching movements from neural activity. We will compare their performance in terms of accuracy, training stability, and the representations they learn through post hoc analysis.''')


    st.write("## Navigation")
    st.write("In this work, we have five different sections. EAch are listed and described below.")

    clmns = st.columns(5)

    clmns_out = st.columns(2)

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
            st.write("## Background ")
            st.write("We will discuss the simple motivations of the project that include")
            st.write('''
                    - Neural Decoding Tasks
                    - Transformers
                    - Hopfield Networks.
                     ''')

        if st.button("Go To Page", key = "background"):
            st.switch_page("pages/background.py")


        with clmns_out[1]:
            st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/real_images/image.png")

    if st.session_state.active_section == "experiments":

        with clmns_out[0]:
            st.write("## Experiments")
            st.write("We will discuss our experiments and components such as:")
            st.write('''
                    - Datasets
                    - Models
                    - Evaluation and Post Analysis
                     ''')

        if st.button("Go To Page", key="experiments"):
            st.switch_page("pages/experiments_and_design.py")


        with clmns_out[1]:
            st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/real_images/hopfield_network.png")

    if st.session_state.active_section == "results":

        with clmns_out[0]:
            st.write("## Summary ")
            st.write("We will provide a summary of our work, including")
            st.write('''
                    - Validation Scores
                    - Performance Trends
                    - Observations from Graphs
                     ''')

        if st.button("Go To Page", key="summary"):
            st.switch_page("pages/training_performance.py")


        with clmns_out[1]:
            st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/real_images/results_for_model.png")

    if st.session_state.active_section == "xai":

        with clmns_out[0]:
            st.write("## Experiments")
            st.write("We will discuss our experiments and components such as:")
            st.write('''
                    - Datasets
                    - Models
                    - Evaluation and Post Analysis
                     ''')

        if st.button("Go To Page", key="xai"):
            st.switch_page("pages/explainability.py")


        with clmns_out[1]:
            st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/real_images/hopfield_network.png")

    if st.session_state.active_section == "summary":

        with clmns_out[0]:
            st.write("## Summary ")
            st.write("We will provide a summary of our work, including")
            st.write('''
                    - Main Results
                    - Further Work and Conclusions
                    - Limitations of Work
                     ''')
        #go_to_bt = st.button("Go To Page")

        if st.button("Go To Page", key="summary"):
            st.switch_page("pages/conclusions.py")


        with clmns_out[1]:
            st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/real_images/results_for_model.png")
