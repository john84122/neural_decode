# Description of Code

In this markdown file, we will lay out what are the basic components that make up the git repository. There are two main levels we will discuss in our code base.

Some notes is that initially much of this code is writtein in Jupyter notebooks. These notebooks can be found in the additional_scripts/full_pipeline directory. However, we have reorganized the code in the src directory of the neural_comp_project git repo. The hope is that the organization of the scripts match exactly what we discuss in the design document.

For the next two sections, we will go over the partitions of the directory (*high level*) as well as the code itself (*Source code*)

## High Level Directory Description

At a high level, we follow the document design pipeline to partition our code. This is shown in the Figure Below (If not seeing this in our git repo, please see the results/misc directory).

![Document Design Object](results/misc/document_design_img.png)

There are five different directories which we will briefly go over:

1. **dataset**

- This is the code used to Load the dataset as well as contains helper functions for augmentations and tokenization of the temporal data.

2. **models**

- This directory contains the scripts with the model architectures. Currently we have five different scripts, but only use the hopfield_only.py, trainsformer.py, and transformer_hopfield.py scripts for this project.

3. **training**

- This directory contains the main training script for training all three of the models we attempt to anlayze.

4. **evaluation**

- The evaluation directory contains all graphing and metric functions we call to evaluate oru models.

5. **post_hoc_analysis**

- This directory contains methods of collecting/producing lower dimensions embeddings from these models as well as producing saliency maps.

## The Source Code

For each source file, we will go over the main scripts in each of these cases. While quite simple, we will discuss what functionalities are in each of these components.

1. **dataset**

> **helper_functions.py**

There are five functions that help with the dataset in this script.

*generate_sinusoidal_position_embeddings*

* This script takes in the specified number of timesteps in our temporal dataset and applies a the sine and cosine functions to the product of position and div_term in order to produce positional ebeddings that the models use.

* For our dataset,the number of timesteps should be always 100 and input dimension is 55.

*bin_spikes*

* This functions produces the input of the model and forms it into a 2D array. The main concept is that how we intially tokenize our data is that we are given inputs of spikes. To convert these values into time steps, we define a timestep that is considered one dimension of our input to a transformer, count for each unit the number of spikes for a given timestep, and that forms a bin that is inputted.

* In this function, spikes are the spiking data, num_units are the dimensions of input, and bin_size is the size of the bin (in seconds) that we want to define.

*load_pretrained*

* This code loads a pretrained model. The ckpt_path should be the .pth model for the model.
* The input model is the class initialization of the model we want to load. We do not use it too much in our code, but it is worth having for post hoc analysis.

> **transformer_dataloader.py**

*get_train_val_loaders*

* This script contains the code which creates the validation and train dataloader. These are simple Datasets defined by torch brain and act similar to a simple pytorch dataset.

* Inputs are the recording_id which is the dataset we use (t_20130819_center_out_reach.h5), a configuration file which we generally ignore, and a batch_size which is used for the dataloader and training.

* The function then goes on to first create the train dataset as well as the validation set which can be pulled from the recording_id .h5 file. To define a dataloader, they both give a torch_brain dataset which acts like a pytorch dataset as well as a sampler which is used to define tokens in the recording. The token length (interval that contains a individual instance in the data) is 1.

* Note that, unlike the code in the initial cosyne tutorial, we set number of workers to zero as the number of workers set higher lead to issues on running this code on a mac.

2. **models**

> *hopfield_only.py*

*HopfieldOnlyDecoder*
* This code defines hte class isntance of a hopfield only decoder which only contains hopfield and linear layers (no attention) to do a task.

* The libraries that we use are hflayers in order to define the layers.

* Input i the class is information about hte dataset which includes num_units, bin_size, sequence_length. In addition, it takes in specifics about the models which includes the dimension of the output, the dimension of the hidden layer output in the model, and the number of layers of the model.

* In the __init__ section where we define the model, we have first define two linear layers that are readin (first layer) and a readout layer (last layer). Then, we define the positional_embeddings which is generated using the generate_sinusoidal_position_embs function. Then, finally we define multiple hopfield layers of the model. These hopfield layers consist of a Hopfield module as well as a feed foward module.

* Forward method then defines how a model processes the input using the layers.

* Finally, the tokenize method will define how we augment the data before feeding it into the model. This is done on the dataset where we call the line below:

```python
train_dataset.transform = tf_hopfield_model.tokenize
```
The tokenizer first defines inputs  as the binned data. Then, the output is the velocity target values. What is returned is a dictionary with this data.

> *transformer_hopfield.py*

*TransformerHopfieldDecoder*

* This script defines the class which is our transformer hopfield decoder. It  tkaes in the same type of inputs as the last model described, which include num_units, bin_size, sequence_length, dim_output, dim_hidden, n_layers. However, it takes in n_heads which defines n_heads which is the number of heads associated to each attention layer.

* In the __init__ method, we define a similar readin and readout linear layers that are the first and last layers of the model respectively. Also, the position_embeddings are defined the same as the last models. But, now we define attention layers of the model. This uses Pytorch Implementation of MultiheadAttention which takes in the embedding dimension, numebr of heads defined as an input, and also a batch_first variable. This is combined with a simple feed forward layer. Finally, we define one hopfield layer that is added into the model using the same hflayers python library.

* Next, we define the forward method which defines the order of layers that an input goes through. This is similar to the last model, except we make sure that the input after the positional embedding layer goes through all attention layers. Once it has gone through all layers, then we feed the input through the hopfield layer and finally the readout layer.

* Also, we define a tokenize transform for the dataset. This is defined the same as the last section.

> *transformer.py*

*TransformerNeuralDecoder*

* This code defines the transformer model we use in our experiments. It is defined practically the same way as the other model. the main difference is that we do not add a hopfield layer to the end of the forward method as well as do not define in in the __init__ method of the class.

3. **training**

> *train_model.py*

*train*
* This function is the main function that we use in our code in order to train all three models. this script takes multiple values, including:

  * model - The model we want to train.
  * optimizer - The optimizer used for training the model (usually AdaGrad)
  * train_loader - Our train loader created using functions from the transformer_dataloader.py script.
  * val_loader - The validation dataloader created using the transformer_dataloader.py scirpt.
  * num_epochs - the number of epochs.
  * store_embeddings - Boolean value determining if we shoud save the features.
  * device - device to train on. Default is CPU, but we used TPU in order to train in our work.

* What happens is in the train loop is that for each epoch, we save the validation R^2 score, do a iteration of training across all batches, and then save the loss.

* Afterwards, we compute the validation R^2 performance and then return what the validation loss is.

* The outputs of the models are predictions of the model, true outputs, the R^2 scores collected, and the log_loss from train set collected over time.

*training_step*

* This code defines the training step done during line 32 of the train function. Simple the function zeros out gradients of optimizer, collects predictions, computes the loss, does backpropagation on the model, and then updates the weights.

* Note that the batch["model_inputs"] is used because the batch inputs consist of a dictionary of inputs and output predictions.

4. **evaluation**

> *graphing_functions.py*

*plot_training_curves*

* This simple function plots the R^2 and loss values produced during the training of the models. It uses simple ax plots to plot how the scores change over time. One can define not only the scores you use, but also the title of the plots.

* The graph is sized to be wider and slightly less tall manually in order to visualie the curve to emphasize possible dynamics during training.

*plot_training_curves_wholistic*

* A similar script that does the plotting of both the R^2 validation scores as well as the training loss in one plot.

> *metrics_for_performance*

*move_to_gpu*

* A script that moves the data to a given device.

*r2_score*

* Compute the R^2 correlation value that is used in the train function discussed above. We implement it directly using torch calls rather than using torch given metrics. Note that this is only for a collection of predictions and true outputs. One needs to concatenate these predictions and outputs before using this function.

*Compute_r2*

* Given a dataloader and a model, the function computes the R^2 for the model. Note that it first moves each batch to a specified device (we defualt as CPU) and then it computes the prediction and true values. These values are than appended to a list.

* Once the list are filled with predictions and true values, it computes the R^2 scores using the r2_score function defined above. 

*print_model*

* A simple function that prints the summary and number of parameters of a given input model.

5. **post_hoc_analysis**

> *embedding_analysis.py*

*collect_embeddings*

* This function will collect the embeddings of the last layer of the model. As discussed above, the last layer of the model is generally a linear readout layer. So, what this function does is first defines a forward hook of the model. This hook effectively holds onto the information of the embedding that are fed into the readout layer.

* After collecting the embedding information, what I do is remove the hook, define the embeddings, and then return both the high dimensional embeddings as well as the labels.

*flatten_embeddings_and_labels*

* this code just simply flattens embeddings. The input embeddings should be of dimension (x, 100, 128). What happens in this code is that I want to return a embedding of size (x*100, 128) to produce UMAP embeddings with.

* In addition, what we do is take the norm of the labels which define velocity vectors.

*Compute_umap_embeddings*

* A simple script that computes basic umap embeddings from the flattened embeddings. UMAP is sklearn-esk, in which it fits on the embeddings to learn the embedding space and then transforms the embeddings. The default augmentations are that the embeddongs should be 3 dimensional and we use euclidean distance as our metric.

> *saliency_based_analysis.py*

*get_attribution*

* This script does a simple backwards step to compute a slaiency map for a input dataset and a model.

* A important not is that to compute the gradient, we take the sum of the squared difference between the output of the model and the true labels during the forward pass. This made it easy to compute the gradients with respect to some error.

*return_normalized_and_aggregated_attribution*

* This function normalizes the attribution maps and stacks them into one numpy array. Normalized is done by subtracting the minimum value of the attirbution map from the entire ndarray and then dividing by the max. Values should be between 0 and 1.

*compute_and_return_attribution_maps*

* combines the last two functions into one - computes attribution maps for a given function and dataset and returns the normalized inputs.


## Additional Scripts

- Some jupyter notebooks and quick runs are in the additional_scripts directory. This includes our full pipeline as well as quick_tutorial scripts. Many of the function calls used in these scripts are really just demonstrations of the function calls defined in the src file.

- We specifically designed the code in the quick_tutorial.py and quick_tutorial.ipynb to be fairly clean and hopefully interpretable. Given we give a brief description of these in the README.md file and these are not the main function calls we defined for our project, we will not go into detail here.
