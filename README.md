# Neural Decoding and Equivalent Systems

This project is for understanding the equivalence between models 

## Collaborations

The main collaborators of this project are Elif Ercek and Johannes Bauer

## Dependencies

For this project, we use only Python in order to build our models. The code is written for either Google Colab Notebooks and Anaconda.

For Python, the main python libraries we use are:

- PyTorch
- TorchBrain
- Hopfield Git Repo
- Matplotlib

How to setup and run our code is briefly discussed in the next section.

## Instructions for quick-run script and demo

1. Setting up the Environemnt

**Google Colab Environment Setup**

To run our code, there are two option. The first option is to use jupyter notebooks. If so, the simple commands you need to run in the first cell are:

```python

!pip install hflayers
!pip install git+https://github.com/ml-jku/hopfield-layers
```

This setup can only be used to run the google colab notebokets in the notebook directory. Else, follow the instructions below.

**Command Line Setup**

For this setup, we have two options. The first option is to first setup a conda virtual environment

```bash
conda create -n streamlit_env -python=3.10

conda activate streamlit_env

```

Secondn, we need to download the required python libraryes:

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

2. Conducting a Quick Run of Experiments:

For a quick evaluation of pipeline, see the quick Tutorials in the notebooks directory.

- A simple script for training_and_evaluation: **training_and_evaluation.ipynb**

1. Full Evaluation Pipeline:

For full evaluation, please see the **full_evaluation.py** script.

4. Demo code:

All demo code is within the **demo_code** Directory. To run, do the following on a laptop with some web browser.

```bash

cd demo_code
streamlit run main_application.py

```

## Outputs and Save Files

Output of models and data used will be in the data and results directory. Specifically,

* **data/perich_miller_population_2018** - Directory containing the data used to train our transformers and hopfield networks.
* **results/models** - Directory containing the models we trained (.pth files)
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