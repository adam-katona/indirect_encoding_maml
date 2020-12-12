# Overview
This repository containes the code for the paper: Utilizing the Untapped Potential of Indirect Encoding for Neural Networks with Meta Learning

It contains:
1. Our pytorch implementation of MAML [[1]](#1)
2. Our implementation of hypernetworks [[2]](#2) for fully connected networks.
3. Experiments to compare indirect and direct ecoding when training with meta learning and greedy learning.

Here is a diagram of the indirect architecture used. The yellow boxes are learned parameters, the blue boxes are generated parameters and the gray boxes are functions:

![Alt text](images/architecture.PNG?raw=true "Indirect encoding architecture")

# Instructions
To download and prepare the omniglot dataset run:
```shell
python download_omniglot.py path/to/store/dataset
```
Then replace the path in es_maml/omniglot/omniglot_data_singleton.py with your /path/to/store/dataset
```python
dataset = np.load("/path/to/store/dataset/omniglot.npy")
```

Then set these variables in run_plain_maml_experiemnt.py
```python
RESULT_ROOT_PATH = "/path/to/store/resuts"
EXPERIMENT_NAME = "name_of_result_folder"
gpus = ["cuda:0","cuda:1"]
concurrent_processes_per_gpu = 2
```
Then run the main experiment:
```shell
python run_plain_maml_experiemnt.py
```
This will queue up the runs with all the different configurations, making sure all the gpus are constantly busy.

Results will be saved in the specified RESULT_ROOT_PATH. The run will save various plots, arrays and models. (see plain_maml.py for the details on what is saved)

Results can be analysed with the notebook: analyse_results.ipynb




# Acknowladgment:
The code to download and preprocess the omniglot dataset was taken from https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot.py



## References
<a id="1">[1]</a> 
Finn, Chelsea, Pieter Abbeel, and Sergey Levine. (2017). 
"Model-agnostic meta-learning for fast adaptation of deep networks."
arXiv preprint arXiv:1703.03400

<a id="2">[2]</a> 
Ha, David and Dai, Andrew and Le, Quoc V (2016). 
"Hypernetworks"
arXiv preprint arXiv:1609.09106

