
import numpy as np
import pandas as pd
import streamlit as st


import plotly.express as px

def return_simple_plotly_graph(lst_of_res, labels_for_res, xlabel, ylabel, title):

    dictionary_of_data = {
        "res": [],
        "lb": [],
        "epochs": [],
    }

    for res_lst_spec, lb in zip(lst_of_res, labels_for_res):

        dictionary_of_data["res"] += list(res_lst_spec)
        dictionary_of_data["lb"] += [lb]*len(res_lst_spec)
        dictionary_of_data["epochs"] += list(range(len(res_lst_spec)))

    pd_data = pd.DataFrame(dictionary_of_data)

    fig = px.line(pd_data, x = "epochs", y = "res", color="lb")
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='white', weight = "bold")),
        xaxis_title=dict(text=xlabel, font = dict(size=13, color='white', weight = "bold")),
        yaxis_title=dict(text=ylabel, font = dict(size=13, color='white', weight = "bold")),
)
    return fig



if __name__ == "__main__":

    st.write("# Training Performance")

    st.write("As one can see, we achieve optimal performance on neural decoding tasks. The validation R2 performance and the Training loss during and after training are visualzied below.")

    hopfield_data = pd.read_csv("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/evaluations/hopfield_2.csv")
    tf_hop_data = pd.read_csv("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/evaluations/t_hop_2.csv")
    tf_data = pd.read_csv("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/evaluations/transformer_2.csv")

    cols1, cols2 = st.columns(2)

    with cols1:
        fig_loss = return_simple_plotly_graph([
            hopfield_data["loss"].to_numpy(),
            tf_hop_data["loss"].to_numpy(),
            tf_data["loss"].to_numpy(),
        ],
        ["Hopfield Only",
         "Transformer + Hopfield",
         "Transformer"
         ],
         "Epoch",
         "Train Loss of Model (MSE)",
         "Train Loss of Model over Time"
         )
        
        st.plotly_chart(fig_loss, theme=None)

    with cols2:
        fig_val = return_simple_plotly_graph([
            hopfield_data["r2"].to_numpy(),
            tf_hop_data["r2"].to_numpy(),
            tf_data["r2"].to_numpy(),
        ],
        ["Hopfield Only",
         "Transformer + Hopfield",
         "Transformer"
         ],
         "Epoch",
         "R2 Correlation",
         "Validation R2 of Model over Time"
         )
        
        st.plotly_chart(fig_val, theme = None)

    st.divider()
    st.write("## Specific Evaluation of Examples")
    st.write("With these results, we can show specific examples that compare predictions from the model to predictions of the truth. Specifically, we can prediction positions of a cursor using the velocity. This will produce a path which we can visualize. To visualize examples where models do well and poorly, click the buttons below.")
    
    bc_1, bc_2 = st.columns(2)

    with bc_1:
        bt_path_1 = st.button("Good Predictions")

    with bc_2:
        bt_path_2 = st.button("Bad Predictions")

    if bt_path_1 == True:
        final_pd = pd.read_csv("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/evaluations/good_example_of_pos.csv")

        fig = px.scatter(final_pd, x = "x_pos", y = "y_pos", animation_frame="time", color="label")
        fig.update_traces(marker=dict(size=15))  
        fig.update_xaxes(range=[-4, 4], constrain='domain')
        fig.update_yaxes(range=[-2, 2], scaleanchor='x', scaleratio=1)
        st.plotly_chart(fig)
        st.write("To see traveling, zoom in. Note that we consider this a good prediction because in this case, all points are clearly traveling in the same direction.")

    if bt_path_2 == True:
        final_pd = pd.read_csv("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/evaluations/bad_example_of_pos.csv")

        fig = px.scatter(final_pd, x = "x_pos", y = "y_pos", animation_frame="time", color="label")
        fig.update_traces(marker=dict(size=15))  
        fig.update_xaxes(range=[-5, 5], constrain='domain')
        fig.update_yaxes(range=[-3, 3], scaleanchor='x', scaleratio=1)
        st.plotly_chart(fig)
        st.write("To see traveling, zoom in. Note that we consider this a bad prediction because in this case, all points are traveling in fairly different directions.")


    st.divider()
    st.write("## Main Findings")
    st.write("In all cases, these models perform higher than the 0.5 R2 correlation, where final validation scores are:")
    st.write("- Transformer Model: 0.715")
    st.write("- Hopfield Only Model: 0.711")
    st.write("- Transformer + Hopfield: 0.708")

    st.write("Also note that the loss and teh scores are almost identical. This strongly suggests that the models are very similar to each other. ")

    