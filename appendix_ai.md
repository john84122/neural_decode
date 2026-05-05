## Our Usage of Large Language Models.

AI Tools Used: ChatGPT, Copilate, Claude

For this code, we use a Claude, ChatGPT, and Copilot for fixing and generating code. For the evaluation code in the full_pipeline, we adapted the code from a cosyne tutorial on transformers (https://cosyne-tutorial-2025.github.io - notebook 2). In order to build hopfield models and the hopfield + transformer models, we use claude and chatgpt in order to debug the architecture we built. In addition, we used AI to find out how to install the right python packages in order to download the code. Finally, we use AI in order to refine documentation on the project code.

For the post hoc analysis and source files with code, we used claude and copilate to edit simple doc strings for the scripts and functions as well as wrote preliminary components of the embedding and saliency analysis. Specifically, it gave a simple function that collects feature embeddings for the post hoc analysis in src/neural_decod/post_hoc_analysis/embedding_analysis.py script. I edited it to make sure that the model was actually getting the correct inputs and returning the desired outputs. In addition, it helped design the Saliency map code in the saliency_based_analysis.py script.

Finally, chatGPT helped desing the logo in the README.md file as well as the start of the pyproject.toml file.