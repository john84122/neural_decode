# Description of Code

In this markdown file, we will lay out the basic components that make up a git repository. There are two places where we provide our main code:

1. additional_scripts -> This is where our full pipeline and quick tutorial code belong.
2. src/neural_decode -> This is where our main scripts and functions that we use are located.

As our main scripts are in the src file, we will mainly discuss the src/neural_decode scripts.

Initially, our code was written in Jupyter notebooks. However, we decided to expand our code in the src files to make it slightly more professional and better suited to a git repo. In the next two subsections, we will go over the partitions of the directory at a (*High Level*) as well as the low level code itself (*Source Code*).

## High Level

At a high level, we follow the document design pipeline to partition our code. This is shown in the figure below (If not seeing this in our git repo, please see the results/misc directory to see the data).

![Document Design Object](results/misc/document_design_img.png)

There are five different directories that partition our code:

1. **dataset**

- This directory contains the scripts to load the dataset, as well as helper functions for augmentations and tokenization of the temporal data.

2. **models**

- This directory contains the scripts with the model architectures. Currently, we have five scripts, but we only use hopfield_only.py, trainsformer.py, and transformer_hopfield.py for this project.

3. **training**

- This directory contains the main training script for training all three of the models we attempt to analyze.

4. **evaluation**

- The evaluation directory contains all graphing and metric functions we call to evaluate our models.

5. **post_hoc_analysis**

- This directory contains methods of collecting/producing lower-dimensional embeddings from these models, as well as producing saliency maps.

## Source Code

For each source file, we will review the main scripts in each case. While each script should be quite simple, we will discuss the functionality of each component.

1. **dataset**

> **helper_functions.py**

This script contains helper functions that do fairly simple operations. Most functions relate back to the dataset.

*generate_sinusoidal_position_embeddings*

* This script takes in the specified number of timesteps in our temporal dataset and applies the sin and cosine functions to the product of position and div_term in order to produce positional embeddings that the models use. This is a transformation discused in the cosyne tutorial for neural decoding and we implement in our methods (tutorial:https://cosyne-tutorial-2025.github.io)

* For our dataset, the number of timesteps should always be 100, and the input dimension is 55.

*bin_spikes*

* This function produces the input tokens to the model and forms them into a 2D array where one dimension is time and the other is the binned inputs at each time step. The main idea is that the raw data we start with consists of sensor spikes. To convert these values into time steps, we define a timestep as one dimension of our transformer input, count the number of spikes for each unit at a given timestep, and form a bin from these counts.

* In this function, spikes are the spiking data, num_units are the dimensions of input, and bin_size is the size of the bin (in seconds) that we want to define.

*load_pretrained*

* This code loads a pretrained model. The ckpt_path should be the .pth model for the model.
* The input model is the class initialization of the model we want to load. We do not use it too much in our code, but it is worth having for post hoc analysis.

> **transformer_dataloader.py**

*get_train_val_loaders*

* This script contains the code that creates the validation and train dataloader. These are simple Datasets defined by TorchBrain and act similarly to a simple PyTorch dataset.

* Inputs are the recording_id which is the dataset we use (t_20130819_center_out_reach.h5), a configuration file which we generally ignore, and a batch_size which is used for the dataloader and training.

* The function then goes on to first create the train dataset as well as the validation set which can be pulled from the recording_id .h5 file. To define a dataloader, they both give a torch_brain dataset, which acts like a PyTorch dataset, as well as a sampler, which is used to define tokens in the recording. The token length (interval that contains an individual instance in the data) is 1.

* Note that, unlike the code in the initial Cosyne tutorial, we set the number of workers to zero as setting the number of workers higher leads to issues when running this code on a Mac.

2. **models**

> *hopfield_only.py*

*HopfieldOnlyDecoder*
* This code defines the class instance of a Hopfield-only decoder, which only contains Hopfield and linear layers (no attention) to do a task.

* The libraries that we use are hflayers in order to define the layers.

* Input in the class is information about the dataset, which includes num_units, bin_size, and sequence_length. In addition, it takes in specifics about the models, which include the dimension of the output, the dimension of the hidden layer output in the model, and the number of layers of the model.

* In the __init__ section where we define the model, we first define two linear layers that are reading (first layer) and a readout layer (last layer). Then, we define the positional_embeddings, which are generated using the generate_sinusoidal_position_embs function. Then, finally, we define multiple Hopfield layers of the model. These Hopfield layers consist of a Hopfield module as well as a feed-forward module.

* The forward method then defines how a model processes the input using the layers.

* Finally, the tokenize method will define how we augment the data before feeding it into the model. This is done on the dataset where we call the line below:

```python
train_dataset.transform = tf_hopfield_model.tokenize
```
The tokenizer first defines inputs as the binned data. Then, the output is the velocity target values. What is returned is a dictionary with this data.

> *transformer_hopfield.py*

*TransformerHopfieldDecoder*

* This script defines the class, which is our transformer Hopfield decoder. It takes in the same type of inputs as the last model described, which include num_units, bin_size, sequence_length, dim_output, dim_hidden, and n_layers. However, it takes in n_heads which defines n_heads which is the number of heads associated to each attention layer.

* In the __init__ method, we define a similar readin and readout linear layers that are the first and last layers of the model respectively. Also, the position_embeddings are defined the same as the last models. But now we define attention layers of the model. This uses Pytorch Implementation of MultiheadAttention which takes in the embedding dimension, numebr of heads defined as an input, and also a batch_first variable. This is combined with a simple feed forward layer. Finally, we define one Hopfield layer that is added into the model using the same hflayers Python library.

* Next, we define the forward method, which defines the order of layers that an input goes through. This is similar to the last model, except we ensure that the input after the positional embedding layer passes through all attention layers. Once it has gone through all these layers, then we feed the input through the Hopfield layer and finally the readout layer.

* Also, we define a tokenize transform for the dataset. This is defined the same as the last section.

> *transformer.py*

*TransformerNeuralDecoder*

* This code defines the transformer model we use in our experiments. It is defined practically the same way as the other model. The main difference is that we do not add a Hopfield layer to the end of the forward method, nor do we define it in the __init__ method of the class.

3. **training**

> *train_model.py*

*train*
* This function is the main function that we use in our code in order to train all three models. This script takes multiple values, including:

  * model - The model we want to train.
  * optimizer - The optimizer used for training the model (usually AdaGrad)
  * train_loader - Our train loader was created using functions from the transformer_dataloader.py script.
  * val_loader - The validation dataloader created using the transformer_dataloader.py scirpt.
  * num_epochs - the number of epochs.
  * store_embeddings - Boolean value determining if we shoud save the features.
  * device - device to train on. Default is CPU, but we used TPU in order to train in our work.

* What happens in the train loop is that for each epoch, we save the validation R^2 score, do an iteration of training across all batches, and then save the loss.

* Afterward, we compute the validation R^2 performance and then return what the validation loss is.

* The outputs of the models are predictions of the model, true outputs, the R^2 scores collected, and the log_loss from train set collected over time.

*training_step*

* This code defines the training step done during line 32 of the train function. Simply, the function zeros the optimizer's gradients, collects predictions and true outputs, computes the loss, performs backpropagation on the model, and then updates the weights.

* Note that the batch["model_inputs"] is used because the batch inputs consist of a dictionary of inputs and output predictions due to the tokenization.

4. **evaluation**

> *graphing_functions.py*

*plot_training_curves*

* This simple function plots the R^2 and loss values produced during the training of the models. It uses simple ax plots to plot how the scores change over time. One can define not only the scores used but also the titles of the plots.

* The graph is sized to be wider and slightly less tall manually in order to visualize the curve to emphasize possible dynamics during training.

*plot_training_curves_wholistic*

* A similar script that does the plotting of both the R^2 validation scores as well as the training loss in one plot.

> *metrics_for_performance*

*move_to_gpu*

* A script that moves the data to a given device.

*r2_score*

* Compute the R^2 correlation value that is used in the train function discussed above. We implement it directly using torch calls rather than using torch given metrics. Note that this is only for a collection of predictions and true outputs. One needs to concatenate these predictions and outputs before using this function.

*Compute_r2*

* Given a dataloader and a model, the function computes the R^2 for the model. Note that it first moves each batch to a specified device (we default to the CPU) and then computes the predictions and true values. These values are then appended to a list.

* Once the list is filled with predictions and true values, it computes the R^2 scores using the r2_score function defined above. 

*print_model*

* A simple function that prints the summary and number of parameters of a given input model.

5. **post_hoc_analysis**

> *embedding_analysis.py*

*collect_embeddings*

* This function will collect the embeddings of the last layer of the model. As discussed above, the model's final layer is typically a linear readout layer. So, this function first defines a forward hook for the model. This hook effectively holds onto the embedding information fed into the readout layer.

* After collecting the embedding information, what I do is remove the hook, define the embeddings, and then return both the high-dimensional embeddings as well as the labels.

*flatten_embeddings_and_labels*

* This code simply flattens embeddings. The input embeddings should have shape (x, 100, 128). What happens in this code is that I want to return an embedding of size (x*100, 128) to produce UMAP embeddings.

* In addition, what we do is take the norm of the labels that define velocity vectors. The idea is that we want to color the embedding space based off the norm of the velocity outputs rather than just one component of the velocity predictions.

*Compute_umap_embeddings*

* A simple script that computes basic UMAP embeddings from the flattened embeddings. UMAP is sklearn-esque: it fits to the embeddings to learn the embedding space and then transforms them. The default augmentations are that the embeddings should be 3-dimensional and we use Euclidean distance as our metric.

> *saliency_based_analysis.py*

*get_attribution*

* This script does a simple backward step to compute a saliency map for an input dataset and a model.

* An important note is that to compute the gradient, we take the sum of the squared difference between the output of the model and the true labels during the forward pass. This made it easy to compute the gradients with respect to some error.

*return_normalized_and_aggregated_attribution*

* This function normalizes the attribution maps and stacks them into one numpy array. Normalized is done by subtracting the minimum value of the attribute map from the entire ndarray and then dividing by the maximum. Values should be between 0 and 1.

*compute_and_return_attribution_maps*

* combines the last two functions into one - computes attribution maps for a given function and dataset, and returns the normalized inputs.


## Additional Scripts

- Some Jupyter notebooks and quick runs are in the additional_scripts directory. This includes our full pipeline as well as quick_tutorial scripts. Many of the function calls used in these scripts are just demonstrations of the functions defined in the src file.

- We specifically designed the code in the quick_tutorial.py and quick_tutorial.ipynb to be fairly clean and hopefully interpretable. Given that we give a brief description of these in the README.md file and these are not the main function calls we defined for our project, we will not go into detail here.
