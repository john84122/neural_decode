# Using Neural Decoding to Establish Architectural Equivalence

This project is to three models on a neural decoding tasks and demonstrate the equivalence between Hopfield Models, Transformers, and a combined Hopfield + Transformer Architecture. We establish this comparison by using their performance on a validation set and differences in Saliency Maps.

## Collaborations

The main collaborators of this project are Elif Ercek and Johannes Bauer

## Dependencies

For this project, we use only Python in order to build our models. The code is written for either Google Colab Notebooks and Anaconda.

Python Version: 3.10

For Python, the main python libraries we use are:

- PyTorch
- TorchBrain
- Hopfield Git Repo
- Matplotlib

Much of the training and evaluation code was done in google colab, but to setup a conda environment to run this code, one can follow the following steps on the command line. First, install conda, if not done so already, and create a new conda environment.

```bash
conda create -n streamlit_env -python=3.10

conda activate streamlit_env

```

Second, we need to download the required python libraryes:

```bash
conda install streamlit
conda install matplotlib

pip install pytorch_brain -q   
pip install git+https://github.com/ml-jku/hopfield-layers
```

Finally, you can setup our git repo.

```bash
git clone https://github.com/john84122/neural_decode.git
pip install -e .
```

When pip installed, you can use the code as a Python Library named neural_decode. A simple example is shown below:

```python
from neural_decode.models.hopfield_only import *

...

```

*Some Additional Notes on Task*
- A common error when installing is the need for a earlier version of numpy. If run into this in the scripts, just run the line in the command line with the virtual environment:

```bash
conda install "numpy<2"
```

- In addition, if this setup is too difficult, please consider using the environment.yaml file for setup.

```bash
 conda env create -f environment.yml
```

## Instructions for quick-run script and demo
1. Conducting a Quick Run of Experiments:

For a quick evaluation of pipeline, see the quick Tutorials in the additional_scripts directory. we have two options for you.

- A Jupyter notebook method: **quick_run.ipynb**

- A .py script for a quick tutorial: **quick_run.py**

The exact results of our analysis can be seen in the results/quick_run_results directory. The output on the command line is in output.txt and also 3 figures.

At a high level, this is what should be shown:

    1. A line saying you are using the cpu.
    2. Lines saying the number of baches in a dataloader as well as last dimension of input.
    3. The architectures of the three loaded models.
    4. Training information for three Epochs of each Model as well as R2 validation score.
    5. Three saved images in the results/quick_run_results directory showing training dynamics curves.
    6. The dimensions of UMAP embeddings.
    7. The dimensions of Saliency maps along with the minimum and maximum values.


2. Full Evaluation Pipeline:

Our full analysis is in the additional_scripts/full_pipeline area of our code.

- For anything due to training and evaluating models, please see the training_and_evaluation.ipynb in google colab. There are already cells in this script to take care of setting up important Python Libraries.

- For post hoc analysis, run the training_and_evaluation.ipynb script.


1. Demo code:

All demo code is within the **demo_code** Directory. To run, do the following on a laptop with some web browser.

```bash

cd demo_code
streamlit run main_application.py

```

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

To download the dataset, type the following in the command line.

```bash
cd ./neuro_comp_project

pip install gdown
gdown 1W--Sm_BcphEC2snoF4zwPdHkkYGgAaUw -O data/perich_miller_population_2018/t_20130819_center_out_reaching.h5
```

This dataset will be downloaded into the data/perich_miller_population_2018 directory.

## Additional Details

- For a good description of the repository, please see the the **description_of_repo.md** markdown file.
- For our policies and usages on Generative LLMs, see our **llm_policy.md** file.