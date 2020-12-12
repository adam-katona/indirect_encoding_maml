


import sys
import numpy as np
import math
import copy
import random
import time

import os
import signal
from datetime import datetime
import subprocess
import uuid

import json
from pydoc import locate
import shutil



def get_default_config():
    default_config = {

        
        "TASK" : "OMNIGLOT",  # SINUSOID_REGRESSION,  OMNIGLOT
        "POLICY_TYPE" : "HYPER_2_LAYER",  # can be DIRECT, HYPER_SIMPLE or HYPER_2_LAYER
        "FULLY_CONNECTED_HIDDEN_DIMS_SINUSOID" : [40,40],  # DO NOT CHANGE, indirect cant handle it for now
        "FULLY_CONNECTED_HIDDEN_DIMS_OMNIGLOT" : [256, 128, 64, 64], # DO NOT CHANGE, indirect cant handle it for now
        "USE_BATCHNORM" : True, # This is both for direct and indirect

        # SETTINGS FOR HYPERNETWORK
        # THIS is for both HYPER_SIMPLE and HYPER_2_LAYER
        "LAYER_1_EMBEDDING_DIM" : 32,
        "LAYER_1_OUT_DIM" : 4,

        # THIS is only for HYPER_2_LAYER and OMNIGLOT
        "LAYER_1_IN_DIM" : 4,    # only used for HYPER_2_LAYER and omniglot
        "LAYER_2_EMBEDDING_DIM" : 32, # only used for omniglot
        "LAYER_2_OUT_DIM" : 4,        # only used for omniglot
        "LAYER_2_IN_DIM" : 4,    # only used for HYPER_2_LAYER and omniglot,

        # For indirect
        "GENERATE_SECOND_LAYER" : True,  # only used for omniglot
        "USE_MIXED_LAYER" : False,   # mix of generated and not generated weights
        "GENERATED_RATIO" : 0.5,

        # if LR_SCHEDULE is used, META_LEARNING_RATE is ignored
        "META_LEARNING_RATE" : 0.001,  
        "META_LR_SCHEDULE_TYPE" : None,  # If None, no schedule else Cosine annealing
        "LR_SCHEDULE_INITIAL_LR" : 0.01,
        "LR_SCHEDULE_RESTART_INTERVAL" : 2000,

        

        "ADAPTATION_LEARNING_RATE" : 0.01, # only used if LEARNABLE_PER_STEP_PER_PARAM_LEARNING_RATE is False
        "NUM_ADAPTATION_GRAD_STEPS" : 1,  

        "LEARNABLE_PER_STEP_PER_PARAM_LEARNING_RATE" : True,
        "INITIAL_ADAPTATION_LEARNING_RATE" : 0.1,   # used in case of USE_PER_STEP_PER_PARAM_LEARNING_RATE is True
        "USE_PER_WEIGHT_LEARNING_RATE" : False,  # This basically dubles the parameters, or more if we use more steps...
        "DPPN_PER_WEIGHT_LEARNING_RATE" : False,  # Use per weight for only DPPN, which have pretty few parameters


        # SUPERVISED META LEARNING PARAMS
        "N_WAY" : 5, # ignored for sinusoidal 
        "SUPPORT_K" : 5, # support set size
        "QUERY_K" : 10, # query set size

        "META_BATCH_SIZE" : 20,
        "EVALUATION_NUM_META_BATCHES" : 6,  # this is just for evaluation, it is not part of learning. This is the number of meta batches,
                                            # so there will be this times META_BATCH_SIZE different sampling of the data.

            }


    return default_config
    

def get_small_net_configs(default_config):

    default_config["TASK"] = "OMNIGLOT"
    default_config["POLICY_TYPE"] = "HYPER_2_LAYER"
    default_config["N_WAY"] = 5
    default_config["SUPPORT_K"] = 1
    default_config["USE_BATCHNORM"] = True
    default_config["TORCH_DEVICE"] = "cuda:1"  # this will be owerwritten
    default_config["NUM_META_GENERATIONS"] = 50000    
    default_config["META_BATCH_SIZE"] = 32   
    default_config["NUM_ADAPTATION_GRAD_STEPS"] = 1  

    default_config["LEARNABLE_PER_STEP_PER_PARAM_LEARNING_RATE"] = True
    default_config["INITIAL_ADAPTATION_LEARNING_RATE"] =  0.01   
    default_config["USE_PER_WEIGHT_LEARNING_RATE"] =  False 
    default_config["GENERATE_SECOND_LAYER"] = True

    default_config["META_LR_SCHEDULE_TYPE"] = "Cosine" # None #"Cosine"
    default_config["LR_SCHEDULE_INITIAL_LR"] = 0.005
    default_config["LR_SCHEDULE_RESTART_INTERVAL"] = 3000

    config_list = []

    # fair indirect encoding, basically the same number of parameters
    net_sizes_and_embedding_dims = [
        ([32,16],[14,2,2],[16,2,2]), # direct 25600  indirect 25748
        ([64,32],[14,4,4],[16,2,2]), # direct 52224  indirect 50784
        ([128,64],[30,4,4],[16,2,2]), # direct 108544  indirect 106328
        #([256, 128],[56,4,4],[32,4,4]),
        ([256, 128, 64, 64],[56,4,4],[32,4,4]), # direct 233472  indirect 233472
    ]


    for net_size,layer_1_dims,layer_2_dims in net_sizes_and_embedding_dims:

        # large direct
        current_config = copy.deepcopy(default_config)
        current_config["POLICY_TYPE"] = "DIRECT"
        current_config["N_WAY"] = 5
        current_config["SUPPORT_K"] = 1
        current_config["FULLY_CONNECTED_HIDDEN_DIMS_OMNIGLOT"] = net_size
        config_list.append(current_config)

        current_config = copy.deepcopy(default_config)
        current_config["POLICY_TYPE"] = "HYPER_2_LAYER"
        current_config["N_WAY"] = 5
        current_config["SUPPORT_K"] = 1
        current_config["FULLY_CONNECTED_HIDDEN_DIMS_OMNIGLOT"] = net_size
        current_config["GENERATE_SECOND_LAYER"] = True
        current_config["LAYER_1_EMBEDDING_DIM"] = layer_1_dims[0]
        current_config["LAYER_1_OUT_DIM"] = layer_1_dims[1]
        current_config["LAYER_1_IN_DIM"] = layer_1_dims[2]   
        current_config["LAYER_2_EMBEDDING_DIM"] = layer_2_dims[0]  
        current_config["LAYER_2_OUT_DIM"] = layer_2_dims[1]       
        current_config["LAYER_2_IN_DIM"] = layer_2_dims[2]   
        config_list.append(current_config)


    # repeat
    #config_list.extend(config_list) # x2
    #config_list.extend(config_list) # x4
    #config_list.extend(config_list) # x8
    #config_list.extend(config_list) # x16

    return config_list


def launch_process_with_config(config,main_experiment_dir,gpu,config_i,src_dir):

    config["TORCH_DEVICE"] = gpu

    # create folder for config
    config_folder_path = os.path.join(main_experiment_dir, "config_" + str(config_i))
    os.makedirs(config_folder_path,exist_ok=True)
    os.chdir(config_folder_path)

    # save job config list
    with open("config.json", 'w') as outfile:
        json.dump(config,outfile, indent=4)

    process = subprocess.Popen(["python " + src_dir + "/run_plain_maml.py"],shell=True)

    print("Process started: ",config_i)
    return process




def distribute_configs_to_gpus(config_list,concurrent_processes_per_gpu,gpus,main_experiment_dir,src_dir):

    config_i = 0
    running_processes_gpu_lists = []
    for gpu in gpus:
        running_processes_gpu_lists.append([])

    while True:

        # check if any running process is done, if yes remove them from the list
        for gpu_i,gpu in enumerate(gpus):

            updated_running_processes = []
            for p in running_processes_gpu_lists[gpu_i]:
                poll = p.poll()
                if poll == None:
                    # p.subprocess is alive
                    updated_running_processes.append(p)
                else:
                    print("Process done")
            running_processes_gpu_lists[gpu_i] = updated_running_processes


        # if there are configs left, try to find free spots
        configs_lefts = len(config_list)-1-config_i
        if configs_lefts > 0:
            for gpu_i,gpu in enumerate(gpus):
                if len(running_processes_gpu_lists[gpu_i]) < concurrent_processes_per_gpu:
                    # we found a space, put the process here 
                    current_config = config_list[config_i]
                    config_i += 1
                    process = launch_process_with_config(config=current_config,main_experiment_dir=main_experiment_dir,
                                                         gpu=gpu,config_i=config_i,src_dir=src_dir)
                    running_processes_gpu_lists[gpu_i].append(process)


        else:
            # no configs left, check if there is any running process left
            num_running_processes = [len(l) for l in running_processes_gpu_lists]
            num_running_processes_total = sum(num_running_processes)
            if num_running_processes_total < 1:
                print("Config list empty, all process done!")
                break
                
        time.sleep(0.05)




if __name__ == '__main__':

    RESULT_ROOT_PATH = "/home/userfs/a/ak1774/workspace/esmaml_main/maml_runs"
    EXPERIMENT_NAME = "SMALL_NETS"
    gpus = ["cuda:0","cuda:1"]
    concurrent_processes_per_gpu=2

    SRC_DIR = os.path.dirname(os.path.abspath(__file__))

    default_config = get_default_config()

    # genrate config list
    config_list = get_small_net_configs(default_config)
    
    # create experiment folder
    main_experiment_dir_name = EXPERIMENT_NAME + datetime.now().strftime("_%m_%d___%H:%M")
    main_experiment_dir = os.path.join(RESULT_ROOT_PATH, main_experiment_dir_name)
    os.makedirs(main_experiment_dir,exist_ok=True)
    os.chdir(main_experiment_dir)

    print("Queuing up " + str(len(config_list)) + " runs. Result will be saved in " + main_experiment_dir)

    distribute_configs_to_gpus(config_list=config_list,
                               concurrent_processes_per_gpu=concurrent_processes_per_gpu,
                               gpus=gpus,
                               main_experiment_dir=main_experiment_dir,
                               src_dir=SRC_DIR)
    

