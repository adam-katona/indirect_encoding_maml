
from pydoc import locate
import sys
import numpy as np
import math
import copy
import random
import time
import torch
import matplotlib.pyplot as plt

import itertools

import es_maml
from es_maml import policies
from es_maml import tasks




def create_per_parameter_learning_rates(net,config):

    # each parameter have its own learning rate for each adaptation step
    # dict of list:  {parameter_name : per_step_learning_rates}
    # per_step_learning_rates can have the shape [n_steps] or [n_steps,*param_shape]
    # depending on weather is a single learning rate for the whole layer, or it is a per weight learning rate
    per_param_per_step_learning_rates = torch.nn.ParameterDict()

    # careful, dont iterate over model.named_parameters(), but get a copy beforehand and iterate on that
    # as we define new parameters torch will update its parameter list, which maybe modifies the dict we are iterating over.
    named_params = dict(net.named_parameters())
    for name,p in named_params.items():
        name = name.replace(".","DOT") # torch parameter name (even parameter dict key ) cant contain "."
        if((config["USE_PER_WEIGHT_LEARNING_RATE"] is True) or
          ((config["DPPN_PER_WEIGHT_LEARNING_RATE"] is True) and "dppn" in name.lower())):
            per_param_per_step_learning_rates[name] = torch.nn.Parameter(torch.ones(config["NUM_ADAPTATION_GRAD_STEPS"],*(p.shape)) * 
                                                                                config["INITIAL_ADAPTATION_LEARNING_RATE"])

        else:
            per_param_per_step_learning_rates[name] = torch.nn.Parameter(torch.ones(config["NUM_ADAPTATION_GRAD_STEPS"]) * 
                                                                            config["INITIAL_ADAPTATION_LEARNING_RATE"])

    return per_param_per_step_learning_rates




# samples a meta batch, takes gradient steps (without owerwriting model params) 
# and calculates adapted losses
# is_test determines weather to use train or eval tasks
# This function needs refactoring, it was originaly a nested function, then I realized i need to call it from elsewhare...
def evaluate_meta_batch(net,per_param_per_step_lrs,config,device,is_test,record_adaptaion_curve=False):
    results = {}

    loss_fn = tasks.get_loss_function(config)
    sample_meta_batch_fn = tasks.get_sample_meta_batch_fn(config)

    NEED_ACCURACY = config["TASK"] == "OMNIGLOT"

    n_way = config["N_WAY"]
    if config["TASK"] == "SINUSOID_REGRESSION":
        n_way = 1


    meta_batch_size = config["META_BATCH_SIZE"]
    if is_test is True:
        meta_batch_size = config["EVALUATION_NUM_META_BATCHES"] * config["META_BATCH_SIZE"]

    support_x,support_y,query_x,query_y = sample_meta_batch_fn(
                                            meta_batch_size=meta_batch_size,
                                            n_way=n_way,
                                            support_k=config["SUPPORT_K"],
                                            query_k=config["QUERY_K"],
                                            is_test=is_test)
    if torch.is_tensor(support_x) is False:
        support_x = torch.from_numpy(support_x)
        query_x = torch.from_numpy(query_x)
        support_y = torch.from_numpy(support_y)
        query_y = torch.from_numpy(query_y)

    support_x = support_x.to(device)
    support_y = support_y.to(device)
    query_x = query_x.to(device)
    query_y = query_y.to(device)                        


    # after the last adaptation step
    meta_batch_adapted_test_losses = []
    meta_batch_adapted_test_accuracies = []

    num_adaptation_steps = config["NUM_ADAPTATION_GRAD_STEPS"]
    if record_adaptaion_curve is True:
        num_adaptation_steps = 3  # only do 3 steps, because it uses a lot of memory
        adaptation_curve_losses = []
        adaptation_curve_accuracies = []
        for _ in range(num_adaptation_steps+1):
            adaptation_curve_losses.append([])
            adaptation_curve_accuracies.append([])



    for task_i in range(support_x.shape[0]):

        # initialize current_params with the net parameters (for the first adaptaion step)
        current_params = dict(net.named_parameters())   
        
        for adaptaion_i in range(num_adaptation_steps):


            out = net(support_x[task_i],custom_params=current_params)
            loss = loss_fn(out,support_y[task_i])

            # Before taking the gradient step calculate the test losses if we are recording the adaptation curve
            # For this we use the query set, and we do not want gradients
            # Be carefull not to owerwrite loss, we still need it
            if record_adaptaion_curve is True:
                with torch.no_grad(): # no grad for test losses
                    out = net(query_x[task_i],custom_params=current_params)
                    intermediate_test_loss = loss_fn(out,query_y[task_i])
                    adaptation_curve_losses[adaptaion_i].append(intermediate_test_loss)
                    if NEED_ACCURACY is True:
                        pred_q = torch.nn.functional.softmax(out, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, query_y[task_i]).sum().item()
                        adaptation_curve_accuracies[adaptaion_i].append(float(correct)/pred_q.numel())


            # take a gradient step, and store the result in adapted_params
            # Here we actually need retain_graph, since we want to differentiate through 
            # the same weight generation for each task.
            # Now we have to be extra careful to call net.after_weight_update(), because
            # it will not be an error now.
            gradients = torch.autograd.grad(loss, current_params.values() ,retain_graph=True)
            

            adapted_params = {}
            for grad,(name,param) in zip(gradients, current_params.items()):

                param_lr = config["ADAPTATION_LEARNING_RATE"]
                if config["LEARNABLE_PER_STEP_PER_PARAM_LEARNING_RATE"] is True: 
                    # In case of evaluation, we might do more adaptation steps than during training
                    # If this is the case, use the last learning rate for the rest of the steps
                    lookup_name = name.replace(".","DOT") # torch parameter name (parameter dict key ) cant contain "."
                    lr_i = adaptaion_i
                    if adaptaion_i >= config["NUM_ADAPTATION_GRAD_STEPS"]:
                        lr_i = config["NUM_ADAPTATION_GRAD_STEPS"]-1
                    param_lr = per_param_per_step_lrs[lookup_name][lr_i]  
                    # param_lr can be a scalar or have the shape of param,
                    # if it is a scalar, that is fine, it is like a normal learning rate
                    # if it has the shape of param that is fine as well, we do an elementwise multiply

                modified_param = param - param_lr * grad
                adapted_params[name] = modified_param

            current_params = adapted_params # prepare for next iteration

        # after the last gradient step
        # calculate the adapted test loss
        out = net(query_x[task_i],custom_params=current_params) #,initialize_batch_statistics=False)
        adapted_test_loss = loss_fn(out,query_y[task_i])
        meta_batch_adapted_test_losses.append(adapted_test_loss)

        # also calculate the accuracy
        if NEED_ACCURACY is True:
            with torch.no_grad():
                pred_q = torch.nn.functional.softmax(out, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, query_y[task_i]).sum().item()
                meta_batch_adapted_test_accuracies.append(float(correct)/pred_q.numel())

        # also append the adapted test losses to the adaptation curve
        if record_adaptaion_curve is True:
            adaptation_curve_losses[num_adaptation_steps].append(adapted_test_loss.detach()) 
            if NEED_ACCURACY is True:
                adaptation_curve_accuracies[num_adaptation_steps].append(float(correct)/pred_q.numel())



    # join the test losses from each task
    meta_batch_adapted_test_losses = torch.stack(meta_batch_adapted_test_losses)
    mean_adapted_losses = torch.mean(meta_batch_adapted_test_losses)


    results["mean_adapted_losses"] = mean_adapted_losses
    results["meta_batch_adapted_test_losses"] = meta_batch_adapted_test_losses
    if NEED_ACCURACY is True:
        results["meta_batch_adapted_test_accuracies"] = meta_batch_adapted_test_accuracies
    if record_adaptaion_curve is True:
        results["adaptation_curve_losses"] = adaptation_curve_losses
        if NEED_ACCURACY is True:
            results["adaptation_curve_accuracies"] = adaptation_curve_accuracies

    return results





# When this function is called, we are already in the run folder, we can just dump the logs at the current directory
def run_maml(config):

    
    device = torch.device(config["TORCH_DEVICE"])
    NEED_ACCURACY = config["TASK"] == "OMNIGLOT"

    loss_fn = tasks.get_loss_function(config)
    sample_meta_batch_fn = tasks.get_sample_meta_batch_fn(config)


    net = policies.create_policy(config,device)
    optimizable_param_list = list(net.parameters())


    if config["LEARNABLE_PER_STEP_PER_PARAM_LEARNING_RATE"] is True:
        per_param_per_step_lrs = create_per_parameter_learning_rates(net,config)
        per_param_per_step_lrs = per_param_per_step_lrs.to(device)
        optimizable_param_list.extend(list(per_param_per_step_lrs.parameters()))

    
    if config["META_LR_SCHEDULE_TYPE"] is None:
        meta_optim = torch.optim.Adam(optimizable_param_list, lr=config["META_LEARNING_RATE"])
    else:
        meta_optim = torch.optim.Adam(optimizable_param_list, lr=config["LR_SCHEDULE_INITIAL_LR"])
        lr_schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(meta_optim,T_0=config["LR_SCHEDULE_RESTART_INTERVAL"], T_mult=1, eta_min=0,)



    gen_adapted_test_losses = []
    gen_adapted_test_accuracies = []
    lr_schedule_log = []

    eval_adapted_test_losses = []
    eval_adapted_test_accuracies = []
    eval_plots_x_axis = []

    experiment_start_time = time.time()
    for gen_i in range(config["NUM_META_GENERATIONS"]):

        meta_batch_results = evaluate_meta_batch(net,per_param_per_step_lrs,config,device,is_test=False)
        
        mean_adapted_losses = meta_batch_results["mean_adapted_losses"]

        meta_optim.zero_grad()
        mean_adapted_losses.backward()
        meta_optim.step()
        if config["META_LR_SCHEDULE_TYPE"] is not None:
            lr_schedule.step()
            lr_schedule_log.append(lr_schedule.get_last_lr()[0])
        net.after_weight_update()

        # LOGGING
        gen_adapted_test_losses.append(mean_adapted_losses.detach().cpu().item())
        if NEED_ACCURACY is True:
            mean_adapted_test_accuracy = np.mean(meta_batch_results["meta_batch_adapted_test_accuracies"])
            gen_adapted_test_accuracies.append(mean_adapted_test_accuracy)
        

        if gen_i % 100 == 10:  # eval time is quite large, 12 times as much as a generation. Do it every 100 iterations
            eval_plots_x_axis.append(gen_i)

            eval_start_time = time.time()
            eval_batch_results = evaluate_meta_batch(net,per_param_per_step_lrs,config,device,is_test=True)
            adaptation_curve_results = evaluate_meta_batch(net,per_param_per_step_lrs,config,device,is_test=True,record_adaptaion_curve=True)
            eval_elapsed = time.time() - eval_start_time

            # log eval stats
            eval_adapted_test_losses.append(eval_batch_results["mean_adapted_losses"].detach().cpu().item())
            if NEED_ACCURACY is True:
                eval_mean_adapted_test_accuracies = np.mean(eval_batch_results["meta_batch_adapted_test_accuracies"])
                eval_adapted_test_accuracies.append(eval_mean_adapted_test_accuracies)

            # log adaptation curve
            adaptation_curve_losses = adaptation_curve_results["adaptation_curve_losses"]
            adaptation_curve_losses = np.mean(adaptation_curve_losses,axis=1)
            if NEED_ACCURACY is True:
                adaptation_curve_accuracies = adaptation_curve_results["adaptation_curve_accuracies"]
                adaptation_curve_accuracies = np.mean(adaptation_curve_accuracies,axis=1)


            generation_time = (time.time() - experiment_start_time) / gen_i
            print(gen_i,generation_time,eval_elapsed,gen_adapted_test_losses[-1])
            plot_x_axis = list(range(len(gen_adapted_test_losses)))
            plt.plot(plot_x_axis,gen_adapted_test_losses)
            plt.plot(eval_plots_x_axis,eval_adapted_test_losses)
            plt.xlabel("meta updates")
            plt.ylabel("loss")
            plt.legend(["loss","eval_loss"])
            plt.savefig("learning_curves.png")
            plt.clf()

            np.save("gen_adapted_test_losses",gen_adapted_test_losses)
            np.save("eval_adapted_test_losses",eval_adapted_test_losses)

            if NEED_ACCURACY is True:
                plt.plot(plot_x_axis,gen_adapted_test_accuracies)
                plt.plot(eval_plots_x_axis,eval_adapted_test_accuracies)
                plt.xlabel("meta updates")
                plt.ylabel("accuracy")
                plt.legend(["accuracy","eval_accuracy"])
                plt.savefig("accuracy_curves.png")
                plt.clf()

                np.save("gen_adapted_test_accuracies",gen_adapted_test_accuracies)
                np.save("eval_adapted_test_accuracies",eval_adapted_test_accuracies)

            plt.plot(adaptation_curve_losses)
            plt.xlabel("adaptation_steps")
            plt.ylabel("loss")
            plt.savefig("adaptation_curve.png")
            plt.clf()
            if NEED_ACCURACY is True:
                plt.plot(adaptation_curve_accuracies)
                plt.xlabel("adaptation_steps")
                plt.ylabel("accuracy")
                plt.savefig("adaptation_accuarcy_curve.png")
                plt.clf()

            np.save("adaptation_curve_losses",adaptation_curve_losses)
            if NEED_ACCURACY is True:
                np.save("adaptation_curve_accuracies",adaptation_curve_accuracies)
            
            # save checkpoint
            if config["META_LR_SCHEDULE_TYPE"] is not None:
                plt.plot(lr_schedule_log)
                plt.savefig("lr_schedule_log.png")
                plt.clf()

                if config["LR_SCHEDULE_RESTART_INTERVAL"] - (gen_i % config["LR_SCHEDULE_RESTART_INTERVAL"]) < 100:
                    # save the model when the learning rate is low
                    # this is so we have a model which is good, and not after warm restart
                    torch.save(net.get_theta().cpu().data , 'theta.torch')
                    torch.save(per_param_per_step_lrs, "lrs.torch")
                else:
                    torch.save(net.get_theta().cpu().data , 'theta_latest.torch')
                    torch.save(per_param_per_step_lrs, "lrs_latest.torch")
            else:
                torch.save(net.get_theta().cpu().data , 'theta.torch')
                torch.save(per_param_per_step_lrs, "lrs.torch")  # per_param_per_step_lrs is a param_dict





        


