# Using Neural Decoding to Establish Architectural Equivalence

![Our Logo](results/misc/icon_for_git_page.png)

This project is to three models on a neural decoding tasks and demonstrate the equivalence between Hopfield Models, Transformers, and a combined Hopfield + Transformer Architecture. We establish this comparison by using their performance on a neural decoding datasets and differences in Saliency Maps.

## Collaborations

The main collaborators of this project are Elif Ercek and Johannes Bauer.

## Dependencies

For this project, we use only Python in order to build our models. The code is written for either Google Colab Notebooks and Anaconda.

Python Version: 3.10

For Python, the main python libraries we use are:

- PyTorch
- TorchBrain
- Hopfield Git Repo
- Matplotlib
- streamlit

Much of the training and evaluation code was done in google colab, but to setup a conda environment to run this code, one can follow the following steps on the command line (also listed in the configuration.txt file). First, install conda, if not done so already, and create a new conda environment.

```bash
conda create -n neural_decode_env python=3.10
conda activate neural_decode_env

```

Second, we need to download the required python libraries:

```bash
conda install -c conda-forge "numpy<2"

conda install -c conda-forge matplotlib umap-learn plotly

pip install streamlit streamlit-webrtc numpy pandas
pip install pytorch_brain
pip install git+https://github.com/ml-jku/hopfield-layers
```

Finally, you can setup our git repo.

```bash
git clone https://github.com/john84122/neural_decode.git
pip install -e .

conda install numba=0.60.0 numpy=1.26.4
```

When pip installed, you can use the code as a Python Library named neural_decode. A simple example of what you would write in a python script is shown below:

```python
from neural_decode.models.hopfield_only import *

...

```

*Some Additional Notes on Task*
- A common error when installing is the need for a earlier version of numpy. If run into this in the scripts, just run the line in the command line with the virtual environment:

```bash
conda install "numpy<2"
```

- The conda environment is very sensitive and slgiht changes to python libraries do lead to problems in the quick_run.py script. Note that when you do setup the conda environment, there may be problems with specifically versions of numpy causing issues wiht UMAP and Saliency Map to run for very long periods of time.

- Because of this, we have removed the UMAP experimentation in the quick_run.py and .ipynb scripts. If one wants to run umap analysis, one may have to collect embeddings using one environment and using another environment to produce the UMAP embeddings.

- However, we do have a working environment to run this code whose packages are shown in the environment.yaml file.

## Instructions for quick-run script and demo
### Conducting a Quick Run of Experiments

For a quick evaluation of pipeline, see the quick Tutorials in the <mark>additional_scripts/quick_tutorials</mark> directory. we have two options for you.

- A Jupyter notebook method: **quick_run.ipynb**

    - This script should be most likely slower as you must run each cell, but it allows you to debug what is going wrong in the quick_run.py script.

- A .py script for a quick tutorial: **quick_run.py**

    - This is the quick_run that we used. Note that an ugly warning occurs at the end of running the script. However, we believe it is safe to ignore as it does not seem to indicate any of our code is incorrect.

The exact results of our analysis can be seen in the results/quick_run_results directory. The output on the command line is in output.txt and also 3 figures.

At a high level, this is what should be shown:

    1. A line saying you are using the cpu.
    2. Lines saying the number of baches in a dataloader as well as last dimension of input.
    3. The architectures of the three loaded models.
    4. Training information for three Epochs of each Model as well as R2 validation score.
    5. Three saved images in the results/quick_run_results directory showing training dynamics curves.
    6. The dimensions of UMAP embeddings.
    7. The dimensions of Saliency maps along with the minimum and maximum values.

### Full Evaluation Pipeline

Our full analysis is in the additional_scripts/full_pipeline area of our code. All of these are jupyter notebooks, one of which is expected to be run on google colab. However, all functions called in the notebook should be discuseed briefly in the src scripts.

- For anything due to training and evaluating models, please see the Project_Cognitive.ipynb using google colab. There are already cells in this script to take care of setting up important Python Libraries and very detailed notes on the scripts.

- For post hoc analysis, run the post_hoc_analysis.ipynb script.


### Demo code

All demo code is within the **demo_code** Directory. To run, do the following on a laptop with some web browser.

```bash

cd demo_code
streamlit run main_application.py

```

NOTE: We are not done with the demo as of yet. There will updates on the git repo, but no updates really to the main code. Mostly it will be text and visualizations.

## Outputs and Save Files

Output of models and data used will be in the data and results directory. Specifically,

* **data/perich_miller_population_2018** - Directory containing the data used to train our transformers and hopfield networks.
* **results/models** - Directory containing the models we trained (.pth files)
* * **results/evaluations** - Where all accuracies of models over each epoch were saved.
* **results/embeddings** - Directory containing the embeddings we analyzed.
* **results/explanations** - Directory containing the explanaitions we have collected.

## Links to External Datasets

The links to the datasets we collected is from the torchbrain repository.

    - Link: https://brainsets.readthedocs.io/en/latest/glossary/brainsets.html#perich-miller-population-2018

The data for training should already be there. However, to download the dataset, type the following in the command line.

```bash
cd ./neuro_comp_project

pip install gdown
gdown 1W--Sm_BcphEC2snoF4zwPdHkkYGgAaUw -O data/perich_miller_population_2018/t_20130819_center_out_reaching.h5
```

This dataset will be downloaded into the data/perich_miller_population_2018 directory. It is important the data is in this directory with the data/perich_miller_population_2018 directory.

## Additional Details

- For a good description of the repository, please see the the **code_walkthrough.md** markdown file.
- For our policies and usages on Generative LLMs, see our **appendix_ai.md** file.
- Configuration commands are in **configuration.txt**
