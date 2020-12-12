


import numpy as np
import matplotlib.pyplot as plt
import torch


#################################################
# sinusoid regression, from original MAML paper #
#################################################

# Quote from the paper:
# the amplitude varies within [0.1, 5.0] and the phase varies within [0, π],
# and the input and output both have a dimensionality of 1.
# During training and testing, datapoints x are sampled uniformly from [−5.0, 5.0].
# The loss is the mean-squared error
# between the prediction f(x) and true value. 
# The regressor is a neural network model with 2 hidden layers of size
# 40 with ReLU nonlinearities. When training with MAML,
# we use one gradient update with K = 10 examples with
# a fixed step size α = 0.01, and use Adam as the metaoptimizer"


def sinusoid_get_random_task():
    amplitude = np.random.uniform(0.1, 5.0)
    phase = np.random.uniform(0, np.pi)
    return amplitude,phase

def sinusoid_get_random_task_batch(amplitude,phase,support_k,query_k):
    
    support_x = torch.from_numpy( np.random.uniform(-5.0, 5.0,(support_k,1))).float()
    query_x = torch.from_numpy( np.random.uniform(-5.0, 5.0,(query_k,1))).float()
    
    support_y = amplitude * torch.sin(support_x - phase)
    query_y =  amplitude * torch.sin(query_x - phase)
    
    return support_x,support_y,query_x,query_y

# Will return data in the shape:  [meta_batch_size, k , 1]    
def sinusoid_get_meta_batch(meta_batch_size,n_way,support_k,query_k,is_test=False):
    
    # n_way is always one for sinusoidal regression, so it is ignored here
    # is_test is to differentiate between test and train tasks, for sinusoid regression, we dont differentiate.

    tasks = [sinusoid_get_random_task() for _ in range(meta_batch_size)]
    
    support_x,support_y,query_x,query_y = [],[],[],[]
    for amplitude,phase in tasks:
        a,b,c,d = sinusoid_get_random_task_batch(amplitude,phase,support_k,query_k)
    
        support_x.append(a)
        support_y.append(b)
        query_x.append(c)
        query_y.append(d)
    
    support_x = torch.stack(support_x)
    support_y = torch.stack(support_y)
    query_x = torch.stack(query_x)
    query_y = torch.stack(query_y)
    
    return support_x,support_y,query_x,query_y


# OMNIGLOT from MAML PAPER
# we also provide results for a non-convolutional network. For this, we use a
# network with 4 hidden layers with sizes 256, 128, 64, 64,
# each including batch normalization and ReLU nonlinearities, followed by a linear layer and softmax. For all models,
# the loss function is the cross-entropy error between the predicted and true class. 
# Additional hyperparameter details are included in Appendix A.1.

# For N-way, K-shot classification, each gradient is computed using a batch size of NK examples. For Omniglot,
# the 5-way convolutional and non-convolutional MAML
# models were each trained with 1 gradient step with step size
# α = 0.4 and a meta batch-size of 32 tasks. The network
# was evaluated using 3 gradient steps with the same step
# size α = 0.4. The 20-way convolutional MAML model
# was trained and evaluated with 5 gradient steps with step
# size α = 0.1. During training, the meta batch-size was set
# to 16 tasks. For MiniImagenet, both models were trained
# using 5 gradient steps of size α = 0.01, and evaluated using
# 10 gradient steps at test time. Following Ravi & Larochelle
# (2017), 15 examples per class were used for evaluating the
# post-update meta-gradient. We used a meta batch-size of
# 4 and 2 tasks for 1-shot and 5-shot training respectively.
# All models were trained for 60000 iterations on a single
# NVIDIA Pascal Titan X GPU

def omniglot_get_meta_batch(meta_batch_size,n_way,support_k,query_k,is_test=False):

    # imort the module the last moment before use, the first import will trigger the load of the whole dataset to memory
    import es_maml.omniglot.omniglot_data_singleton
    data = es_maml.omniglot.omniglot_data_singleton.dataset
    omniglot_shuffled_indicies = es_maml.omniglot.omniglot_data_singleton.omniglot_shuffled_indicies

    SHUFFLE_CLASSES = True
    AUGMENT_WITH_ROTATION = True

    if SHUFFLE_CLASSES is True:
        train_indicies = omniglot_shuffled_indicies[:1200]
        test_indicies = omniglot_shuffled_indicies[1200:]
    else:
        train_indicies = list(range(1200))
        test_indicies = list(range(1200,data.shape[0]))

    class_indicies = train_indicies
    if is_test is True:
        class_indicies = test_indicies

    support_x = []
    query_x = []
    support_y = []
    query_y = []

    for meta_batch_i in range(meta_batch_size):
        selected_class_indicies = np.random.choice(class_indicies,n_way,replace=False)  

        task_support_x = []
        task_query_x = []
        task_support_y = []
        task_query_y = []

        for class_i_in_batch,class_i in enumerate(selected_class_indicies):

            
            selected_images = np.random.choice(list(range(20)),support_k+query_k,replace=False) # if support_k+query_k = 20, this will be a permutation
            
            class_data = data[class_i,selected_images]

            # Each class can be augmented by rotation, we select the rotation after selecting distinct classes
            # This means we cannot have a task with the same charater with different rotations, which is what we want
            if AUGMENT_WITH_ROTATION is True:
                selected_rotation = np.random.choice([0,1,2,3]) # multiples of 90 degree
                # np.rot90 cannot handle channels, take the one channel, channel 0, and add it back after rotation
                class_data = [np.rot90(class_data[i,0],selected_rotation).reshape(1,28,28) for i in range(len(selected_images))]
                class_data = np.stack(class_data) # we are back to the original shape 

            class_support_x = class_data[:support_k]
            class_query_x = class_data[support_k:]

            class_support_y = np.repeat(class_i_in_batch,support_k)
            class_query_y = np.repeat(class_i_in_batch,query_k)

            task_support_x.append(class_support_x)
            task_query_x.append(class_query_x)
            task_support_y.append(class_support_y)
            task_query_y.append(class_query_y)

        task_support_x = np.stack(task_support_x)
        task_query_x = np.stack(task_query_x)
        task_support_y = np.stack(task_support_y)
        task_query_y = np.stack(task_query_y)

        support_x.append(task_support_x)
        query_x.append(task_query_x)
        support_y.append(task_support_y)
        query_y.append(task_query_y)
    
    support_x = np.stack(support_x)
    query_x = np.stack(query_x)
    support_y = np.stack(support_y)
    query_y = np.stack(query_y)

    # reshape to: meta batch size, batch size, input_size
    support_x = support_x.reshape((meta_batch_size,n_way*support_k,1*28*28))
    query_x = query_x.reshape((meta_batch_size,n_way*query_k,1*28*28))
    support_y = support_y.reshape((meta_batch_size,n_way*support_k))
    query_y = query_y.reshape((meta_batch_size,n_way*query_k))

    return support_x,support_y,query_x,query_y








def get_sample_meta_batch_fn(config):
    if config["TASK"] == "SINUSOID_REGRESSION":
        return sinusoid_get_meta_batch
    elif config["TASK"] == "OMNIGLOT":
        return omniglot_get_meta_batch
    else:
         raise "Unknown Task!!"

def get_loss_function(config):
    if config["TASK"] == "SINUSOID_REGRESSION":
        return torch.nn.functional.mse_loss
    elif config["TASK"] == "OMNIGLOT":
        return torch.nn.functional.cross_entropy
    else:
         raise "Unknown Task!!"





