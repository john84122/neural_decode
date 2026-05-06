import streamlit as st


def main():
    st.write("# Summary")
    st.write("## Conclusion of Our Results")
    st.write("In our experiments, we used three different architectures to learn a neural decoding task. We sought to answer two specific questions:")
    st.write("  - How equivalent are Transformer Models to Hopfield Networks in their performance on Neural Decoding Tasks?")
    st.write("  - Are transformers and Hopfield Networks good for Neural Decoding?")

    st.write()
    st.write("For the first question, we find sufficient evidence that Hopfield and Transformer Models are fairly similar to each. This is seen in both the similarities in accuracies and embedding spaces. While we do do t-tests on the differences between attribution maps and find that differences between attribution maps is significnatly different from zero, there are deep limitations to the methods we use in this case. We wil address this in the limitations section.")
    st.write()
    st.write("For the second question, we find that both transformers and hopfield networks are good for neural decoding. Their R2 validation was far beyond the 0.5 correlation score we wanted these models to beat.")


    st.divider()
    st.write("## Limitatations")

    st.write("While we make claims that there are similarities between these architectures, there are limitations to how strong we can claim this similarity is. Our first claim was that these models are similar because they achieve the same accuracies. While R2 is a good measure of performance, note that there is more than one way to do a task. So, these models can be very different in how they approach a task and at the same time do well on predicting velocity. We attempt to address this limitation using post hoc analysis. However, methods such as UMAP and saliency maps are imperfect descriptions of what a model uses for the neural decoding task and the high dimensional embedding space.")
    st.write()
    st.write("In addition, we did not check our normality assumption for the T-Test. The expectation is that the differences and the number of timesteps we take should be large enough to do a t-test, more work needs to be done to actually show normality. Finally, note that samples are technically not indipendently and identically distributed. These samples come from the same temporal sample. Thus, the t-test results should be taken with a grain of salt.")
    st.write()
    st.write("For our performance evaluation, note that we do not do a very thorough hyperparameter search. Thus, these models might be suboptimal and with more complex methods, we can find models that have much larger validation R2 validation scores.")

    st.divider()
    st.write("## Future Work")

    st.write("For future work, more time should go into more models of a varying architectures. By do a comparison between these models and the three models we chose, we can gain a better understand of how significant effectively these similarities between the three models are.")

    st.divider()
    st.write("## Citations")

    st.write('''
            - Dorkenwald, Sven, et al. "FlyWire: online community for whole-brain connectomics." Nature methods 19.1 (2022): 119-128.
            - McInnes, Leland, John Healy, and James Melville. "Umap: Uniform manifold approximation and projection for dimension reduction." arXiv preprint arXiv:1802.03426 (2018).
            - Ramsauer, Hubert, et al. "Hopfield networks is all you need." arXiv preprint arXiv:2008.02217 (2020).
            - Reisberg, Daniel, and Sheri Snavely. "Cognition: Exploring the science of the mind." (2010).
            - Thakur, Amey, and Archit Konde. "Fundamentals of neural networks." International Journal for Research in Applied Science and Engineering Technology 9.VIII (2021): 407-426.
            - Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
            - Wang, Binxu, and Carlos R. Ponce. "High-performance evolutionary algorithms for online neuron control." Proceedings of the genetic and evolutionary computation conference. 2022.             ''')

if __name__ == "__main__":
    main()