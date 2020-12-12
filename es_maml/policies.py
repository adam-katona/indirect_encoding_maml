


import numpy as np
import matplotlib.pyplot as plt
import torch

def get_shape_info(named_param_dict):
    shape_info = dict()
    current_index = 0 
    for name,param in named_param_dict.items():
        start_end_indicies = (current_index,current_index + param.data.numel())
        current_index += param.data.numel()
        param_shape = param.data.shape
        shape_info[name] = {"indices" : start_end_indicies,"shape" : param_shape }
    return shape_info

def get_params_flat(named_param_dict):
    shape_info = dict()
    current_index = 0 
    param_views = []
    for name,param in named_param_dict.items():
        start_end_indicies = (current_index,current_index + param.data.numel())
        current_index += param.data.numel()
        param_shape = param.data.shape
        shape_info[name] = {"indices" : start_end_indicies,"shape" : param_shape }
        param_views.append(param.data.view(-1))

    return torch.cat(param_views),shape_info  # NOTE the cat will create a copy...

def overwrite_params_with_flat(model,flat_vec,shape_info):
    model_dict = dict(model.named_parameters())
    for key,info in shape_info.items():
        start_i,end_i = info["indices"]     
        model_dict[key].data = flat_vec[start_i:end_i].view(*info["shape"])


##################
# DIRECT NETWORK #
##################


class FunctionalFullyConnectedNet(torch.nn.Module):
    def __init__(self,num_inputs,hidden_dims,num_outputs,config,device=None):         
        super(FunctionalFullyConnectedNet, self).__init__()

        self.device = device
        if self.device is None:
           self.device = torch.device("cpu")


        self.USE_BATCHNORM = False
        if "USE_BATCHNORM" in config:
            self.USE_BATCHNORM = config["USE_BATCHNORM"]

        self.weights = torch.nn.ParameterList()
        self.biases = torch.nn.ParameterList()

        # The parameters for batch norm layers
        # This is for scaling and shifting the already normalized activations (also called gamma and beta)
        self.bn_weights = torch.nn.ParameterList()  
        self.bn_biases = torch.nn.ParameterList()

        # we ar not tracking running statistics, and always use the current batch statistics, same as orig maml paper
        # https://github.com/cbfinn/maml/issues/37

        previous_dim = num_inputs
        for hidden_dim in hidden_dims:
            
            w = torch.nn.Parameter(torch.ones((hidden_dim,previous_dim)))
            torch.nn.init.kaiming_normal_(w)  # init weights to gain 1
            self.weights.append(w)

            b = torch.nn.Parameter(torch.zeros(hidden_dim)) # init biases to zero
            self.biases.append(b)

            if self.USE_BATCHNORM is True:
                self.bn_weights.append(torch.nn.Parameter(torch.ones(hidden_dim)))  
                self.bn_biases.append(torch.nn.Parameter(torch.zeros(hidden_dim)))  
            previous_dim = hidden_dim

        # final layer
        w = torch.nn.Parameter(torch.ones((num_outputs,previous_dim)))
        torch.nn.init.kaiming_normal_(w)  # init weights to gain 1
        self.weights.append(w)

        b = torch.nn.Parameter(torch.zeros(num_outputs)) # init biases to zero
        self.biases.append(b)

        
        self.shape_info = get_shape_info(dict(self.named_parameters()))
        self.to(self.device)


    def forward(self,x,custom_params=None):

        params = dict(self.named_parameters())
        if custom_params is not None:
            params = custom_params

        for layer_i in range(len(self.weights)):
            if layer_i != (len(self.weights)-1): # if not last layer
                
                weights = params["weights."+str(layer_i)] 
                biases  = params["biases."+str(layer_i)] 
                x = torch.nn.functional.linear(x, weights, biases)

                if self.USE_BATCHNORM is True:
                    weights = params["bn_weights."+str(layer_i)]
                    biases = params["bn_biases."+str(layer_i)]

                    # setting training=True to force torch to use the batch statistics, passing None as running statistics.
                    x = torch.nn.functional.batch_norm(x,running_mean=None,# self.bn_running_means[layer_i],
                                                         running_var=None,# self.bn_running_variances[layer_i],
                                                         weight=weights, bias=biases,
                                                         training=True)

                x = torch.relu(x)

            else:
                weights = params["weights."+str(layer_i)] 
                biases  = params["biases."+str(layer_i)]  
                x = torch.nn.functional.linear(x, weights, biases)
        return x
            
    def set_theta(self, theta):
        overwrite_params_with_flat(self,theta,self.shape_info)

    def get_theta(self):
        params_flat,shape_info = get_params_flat(dict(self.named_parameters()))
        return params_flat

    def after_weight_update(self):
        pass # Do absolutlely nothing, just need identical api as indirect net



####################
# INDIRECT NETWORK #
####################

class IndirectNet(torch.nn.Module):
    
    def __init__(self,config,device=None):
        super(IndirectNet, self).__init__()
        
        self.device = device
        if self.device is None:
           self.device = torch.device("cpu")

        self.USE_BATCHNORM = False
        if "USE_BATCHNORM" in config:
            self.USE_BATCHNORM = config["USE_BATCHNORM"]

        self.config = config

        self.weights_generated = False  # flag to show weather we have valid generated weights
        self.generated_weights = None

        # the parameters for not generated layers
        self.not_generated_weights = torch.nn.ParameterList()

        # for each generated layer we have a weights and biases
        self.weights = []  # these are generated, just tensors not torch parameters
        self.weights_shapes = []


        self.biases = torch.nn.ParameterList()

        self.bn_weights = torch.nn.ParameterList()  
        self.bn_biases = torch.nn.ParameterList()
 

        # this is instrunctions for the forward to use which weight for which layer
        self.layer_instructions = []
        # {layer_type,   # can be generated, not_generated, mixed
        #  non_generated_w_i, 
        #  generated_w_i,
        #  bias_i,
        #  bn_params_i, 
        #  use_activation,
        #  generated_ratio,  # ratio of generated weights to non generated, only used when layer_type is mixed
        #  }

        def add_not_generated_layer(from_size,to_size,use_batch_norm,use_activation):
            w = torch.nn.Parameter(torch.ones((to_size,from_size)))
            torch.nn.init.kaiming_normal_(w)  # init weights to gain 1
            b = torch.nn.Parameter(torch.zeros(to_size)) # init biases to zero

            self.not_generated_weights.append(w)
            self.biases.append(b)

            bn_i = None
            if use_batch_norm is True:
                bn_w = torch.nn.Parameter(torch.ones(to_size))  
                bn_b = torch.nn.Parameter(torch.zeros(to_size))  
                self.bn_weights.append(bn_w)  
                self.bn_biases.append(bn_b)
                bn_i = len(self.bn_weights)-1

            self.layer_instructions.append({
                "layer_type" : "not_generated",  #  generated, not_generated, mixed
                "non_generated_w_i" : len(self.not_generated_weights)-1,
                "generated_w_i" : len(self.weights)-1,
                "bias_i" : len(self.biases)-1,
                "bn_params_i" : bn_i,
                "use_activation" : use_activation,
            })


        def add_generated_layer(from_size,to_size,use_batch_norm,use_activation):
           
            self.weights.append(None)
            self.weights_shapes.append((to_size,from_size))
            self.biases.append(torch.nn.Parameter(torch.zeros(to_size)))

            bn_i = None
            if use_batch_norm is True:
                bn_w = torch.nn.Parameter(torch.ones(to_size))  
                bn_b = torch.nn.Parameter(torch.zeros(to_size))
                self.bn_weights.append(bn_w)  
                self.bn_biases.append(bn_b)
                bn_i = len(self.bn_weights)-1

            
            self.layer_instructions.append({
                "layer_type" : "generated",  #  generated, not_generated, mixed
                "generated_w_i" : len(self.weights)-1,
                "bias_i" : len(self.biases)-1,
                "bn_params_i" : bn_i,
                "use_activation" : use_activation,
            })

        # in case of a mixed layer, certain weight are generated, others are directly encoded
        # we still define a full layer for both generated and non generated, and we mix them on forward, and just ingore the rest.
        # they will have 0 grad, and dont change anything.
        def add_mixed_layer(from_size,to_size,use_batch_norm,use_activation,generated_ratio):
            
            # add generated weights
            self.weights.append(None)
            self.weights_shapes.append((to_size,from_size))

            # add non generated weights
            w = torch.nn.Parameter(torch.ones((to_size,from_size)))
            torch.nn.init.kaiming_normal_(w)  # init weights to gain 1
            self.not_generated_weights.append(w)

            # add bias
            self.biases.append(torch.nn.Parameter(torch.zeros(to_size)))

            bn_i = None
            if use_batch_norm is True:
                bn_w = torch.nn.Parameter(torch.ones(to_size))  
                bn_b = torch.nn.Parameter(torch.zeros(to_size))  
                self.bn_weights.append(bn_w)  
                self.bn_biases.append(bn_b)
                bn_i = len(self.bn_weights)-1
        
            self.layer_instructions.append({
                "layer_type" : "mixed",  #  generated, not_generated, mixed
                "non_generated_w_i" : len(self.not_generated_weights)-1,
                "generated_w_i" : len(self.weights)-1,
                "bias_i" : len(self.biases)-1,
                "bn_params_i" : bn_i,
                "use_activation" : use_activation,
                "generated_ratio" : generated_ratio,
            })



        # This can either be for omniglot or sinusoid regression, with different network shapes
        # For SINUSOID_REGRESSION, the networks shape is hardcoded
        if self.config["TASK"] == "SINUSOID_REGRESSION": 
            # layer_dims = [1,40,40,1]
            # The first and the last is not generated
            add_not_generated_layer(from_size=1,to_size=40,use_batch_norm=self.USE_BATCHNORM,use_activation=True)
            if config["USE_MIXED_LAYER"] is True:
                add_mixed_layer(from_size=40,to_size=40,use_batch_norm=self.USE_BATCHNORM,use_activation=True,generated_ratio=config["GENERATED_RATIO"])
            else:
                add_generated_layer(from_size=40,to_size=40,use_batch_norm=self.USE_BATCHNORM,use_activation=True)
            add_not_generated_layer(from_size=40,to_size=1,use_batch_norm=False,use_activation=False) # there is no batch norm in the last layer

        elif self.config["TASK"] == "OMNIGLOT":
            prev_dim = 784
            for layer_i,hidden_dim in enumerate(config["FULLY_CONNECTED_HIDDEN_DIMS_OMNIGLOT"]):

                # the first layer is always generated
                if layer_i == 0:
                    if config["USE_MIXED_LAYER"] is True:
                        add_mixed_layer(from_size=prev_dim,to_size=hidden_dim,use_batch_norm=self.USE_BATCHNORM,use_activation=True,generated_ratio=config["GENERATED_RATIO"])
                    else:
                        add_generated_layer(from_size=prev_dim,to_size=hidden_dim,use_batch_norm=self.USE_BATCHNORM,use_activation=True)


                # the second layer is optionally generated
                elif  layer_i == 1:
                    if config["GENERATE_SECOND_LAYER"] is True:
                        if config["USE_MIXED_LAYER"] is True:
                            add_mixed_layer(from_size=prev_dim,to_size=hidden_dim,use_batch_norm=self.USE_BATCHNORM,use_activation=True,generated_ratio=config["GENERATED_RATIO"])
                        else:
                            add_generated_layer(from_size=prev_dim,to_size=hidden_dim,use_batch_norm=self.USE_BATCHNORM,use_activation=True)
                    else:
                        add_not_generated_layer(from_size=prev_dim,to_size=hidden_dim,use_batch_norm=self.USE_BATCHNORM,use_activation=True)

                # the rest is directly encoded
                else:
                    add_not_generated_layer(from_size=prev_dim,to_size=hidden_dim,use_batch_norm=self.USE_BATCHNORM,use_activation=True)

                prev_dim = hidden_dim
                

            # add final layer
            add_not_generated_layer(from_size=prev_dim,to_size=config["N_WAY"],use_batch_norm=False,use_activation=False)
        
        
        if self.config["POLICY_TYPE"] == "HYPER_SIMPLE" or self.config["POLICY_TYPE"] == "HYPER_2_LAYER":
            if self.config["TASK"] == "SINUSOID_REGRESSION":
                self.generator_net = SinusoidHyperGenerator(self.config,device=self.device)
            elif self.config["TASK"] == "OMNIGLOT":
                self.generator_net = OmniglotHyperGenerator(self.config,device=self.device)
        else:
            raise "Unknown POLICY_TYPE"
        

        self.shape_info = get_shape_info(dict(self.named_parameters()))
        self.to(self.device)
        self.ensure_weights_generated()

        

    def after_weight_update(self):
        self.weights_generated = False  # parameters changed, invalidate generated weights
    

    def generate_weights(self,custom_params=None):
        # if custom_params is None, we use the model parameters and we owerwrite the model generated weights
        # if custom_params is not None, instead of owerwriting the model state, we return the results in a new list.

        if custom_params is None:
            result_list = self.weights  
            custom_params = dict(self.named_parameters())
        else:
            result_list = [None] * len(self.weights_shapes) # prepare an empty list with the right length

        # generate the flat weight vector
        self.generated_weights = self.generator_net(custom_params=custom_params)
  
        # distribute the weights among the weight matricies
        current_i = 0
        for layer_i,layer_shape in enumerate(self.weights_shapes):
            layer_size = layer_shape[0] * layer_shape[1]
            #print("layer_size ",layer_size)
            result_list[layer_i] = self.generated_weights[current_i:current_i+layer_size].reshape(layer_shape)
            current_i = current_i+layer_size

        if custom_params is not None:
            return result_list



    # call this whenever weight are updated
    # This is not inside forward, it should not be recaluclated at every forward call
    # only when the parameters change.
    # It is very important to not forget to call this!!!
    def ensure_weights_generated(self):
        if self.weights_generated is True:
            return
        self.generate_weights() # this will update self.weights
        self.weights_generated = True
    
    
    def forward(self,x,custom_params=None):
        self.ensure_weights_generated()  

        # when called with custom_params, we always force a generation of parameters.
        # Normally you would only generate the weight if the parameters changed, 
        # but with custom params we only ever do 1 batch before we update the weights anyway
        if custom_params is None:
            params = dict(self.named_parameters())
            self.ensure_weights_generated() 
            generated_weights = self.weights
            
        
        else:
            params = custom_params
            generated_weights = self.generate_weights(custom_params=custom_params)


        for layer_instruction in self.layer_instructions:

            # APPLY LINEAR LAYER
            if layer_instruction["layer_type"] == "not_generated":
                weight = params["not_generated_weights."+str(layer_instruction["non_generated_w_i"])]
            elif layer_instruction["layer_type"] == "generated":
                weight = generated_weights[layer_instruction["generated_w_i"]]
            elif layer_instruction["layer_type"] == "mixed":
                weight_ng = params["not_generated_weights."+str(layer_instruction["non_generated_w_i"])]
                weight_g = generated_weights[layer_instruction["generated_w_i"]]
                num_neurons = weight_ng.shape[0]
                num_generated_neurons = int(np.floor( num_neurons * layer_instruction["generated_ratio"]))
                weight = torch.zeros_like(weight_ng)
                weight[:num_generated_neurons] = weight_g[:num_generated_neurons]
                weight[num_generated_neurons:] = weight_ng[num_generated_neurons:]

            else:
                raise "Unknown layer type"

            bias = params["biases."+str(layer_instruction["bias_i"])]
            x = torch.nn.functional.linear(x,weight,bias)

            # APPLY BATCH NORM 
            if layer_instruction["bn_params_i"] is not None:
                bn_weight = params["bn_weights."+str(layer_instruction["bn_params_i"])]
                bn_bias =   params["bn_biases."+str(layer_instruction["bn_params_i"])]
                x = torch.nn.functional.batch_norm(x,running_mean=None, running_var=None, 
                                                     weight=bn_weight, bias=bn_bias, training=True)
                
            # APPLY ACTIVATION
            if layer_instruction["use_activation"]:
                x = torch.relu(x)

        return x
        
        
    def set_theta(self, theta):
        overwrite_params_with_flat(self,theta,self.shape_info)
        self.after_weight_update()

    def get_theta(self):
        params_flat,shape_info = get_params_flat(dict(self.named_parameters()))
        return params_flat
    



#######################################
# GENERATOR MODELS WITH HYPERNETWORKS #
#######################################

# This class contain the hypernetworks which generates parts of the fully connected net for omniglot
# We generate the weight of either 1 or 2 layers.
# There is 1 hypernetwork per layer, each hypernetwork has many embeddings
class OmniglotHyperGenerator(torch.nn.Module):
    def __init__(self,config,device=None):
        super(OmniglotHyperGenerator, self).__init__()

        self.config = config

        self.device = device
        if self.device is None:
           self.device = torch.device("cpu")

        layer_dims = [784] # 28x28=784
        layer_dims.extend(config["FULLY_CONNECTED_HIDDEN_DIMS_OMNIGLOT"])
        layer_dims.append(config["N_WAY"])
        #print(layer_dims)

        if config["POLICY_TYPE"] == "HYPER_SIMPLE":
            # layer_1_hypernetwork generate weight for the layer 784,256
            self.layer_1_hypernetwork = HyperNetworkSimple(z_dim = config["LAYER_1_EMBEDDING_DIM"],
                                                        out_size = config["LAYER_1_OUT_DIM"],
                                                        unit_weight_vec_size = layer_dims[0])
            
            embedding_needed_for_layer_1 = int(np.ceil(float(layer_dims[1]) / config["LAYER_1_OUT_DIM"]))
            self.weights_needed_for_layer_1 = layer_dims[1] * layer_dims[0]

            if config["GENERATE_SECOND_LAYER"] is True:
                # layer_2_hypernetwork generate weight for the layer 256,128
                self.layer_2_hypernetwork = HyperNetworkSimple(z_dim = config["LAYER_2_EMBEDDING_DIM"],
                                                            out_size = config["LAYER_2_OUT_DIM"],
                                                            unit_weight_vec_size = layer_dims[1])
                embedding_needed_for_layer_2 = int(np.ceil(float(layer_dims[2]) / config["LAYER_2_OUT_DIM"]))
                self.weights_needed_for_layer_2 = layer_dims[2] * layer_dims[1]

        elif config["POLICY_TYPE"] == "HYPER_2_LAYER":
            # layer_1_hypernetwork generate weight for the layer 784,256
            self.layer_1_hypernetwork = HyperNetworkSemiTwoLayer(z_dim = config["LAYER_1_EMBEDDING_DIM"],
                                                        in_size = config["LAYER_1_IN_DIM"],
                                                        out_size = config["LAYER_1_OUT_DIM"],
                                                        unit_weight_vec_size = layer_dims[0])
            embedding_needed_for_layer_1 = int(np.ceil(float(layer_dims[1]) / (config["LAYER_1_OUT_DIM"] * config["LAYER_1_IN_DIM"])))
            self.weights_needed_for_layer_1 = layer_dims[1] * layer_dims[0]

            if config["GENERATE_SECOND_LAYER"] is True:
                # layer_2_hypernetwork generate weight for the layer 256,128
                self.layer_2_hypernetwork = HyperNetworkSemiTwoLayer(z_dim = config["LAYER_2_EMBEDDING_DIM"],
                                                            in_size = config["LAYER_2_IN_DIM"],
                                                            out_size = config["LAYER_2_OUT_DIM"],
                                                            unit_weight_vec_size = layer_dims[1])
                embedding_needed_for_layer_2 = int(np.ceil(float(layer_dims[2]) / (config["LAYER_2_OUT_DIM"] * config["LAYER_2_IN_DIM"])))
                self.weights_needed_for_layer_2 = layer_dims[2] * layer_dims[1]

        else:
            raise "Unknown policy type " + config["POLICY_TYPE"]

        

        self.embeddings_layer_1 = torch.nn.ParameterList()
        for i in range(embedding_needed_for_layer_1):
            self.embeddings_layer_1.append(torch.nn.Parameter(torch.randn(config["LAYER_1_EMBEDDING_DIM"])))

        if config["GENERATE_SECOND_LAYER"] is True:
            self.embeddings_layer_2 = torch.nn.ParameterList()
            for i in range(embedding_needed_for_layer_2):
                self.embeddings_layer_2.append(torch.nn.Parameter(torch.randn(config["LAYER_2_EMBEDDING_DIM"])))


    def forward(self,custom_params=None):
        generated_weights_layer_1 = []
        generated_weights_layer_2 = []
        for i,embedding in enumerate(self.embeddings_layer_1):
            current_embedding = embedding
            if custom_params is not None:
                current_embedding = custom_params["generator_net.embeddings_layer_1."+str(i)]
            generated_weights_layer_1.append(self.layer_1_hypernetwork(current_embedding,
                                                               custom_params=custom_params,
                                                               prefix="generator_net.layer_1_hypernetwork").view(-1)) 
        
        if self.config["GENERATE_SECOND_LAYER"] is True:
            for i,embedding in enumerate(self.embeddings_layer_2):
                current_embedding = embedding
                if custom_params is not None:
                    current_embedding = custom_params["generator_net.embeddings_layer_2."+str(i)]
                generated_weights_layer_2.append(self.layer_2_hypernetwork(current_embedding,
                                                                custom_params=custom_params,
                                                                prefix="generator_net.layer_2_hypernetwork").view(-1))

        generated_weights_layer_1 = torch.cat(generated_weights_layer_1)[:self.weights_needed_for_layer_1]
        if self.config["GENERATE_SECOND_LAYER"] is True:
            generated_weights_layer_2 = torch.cat(generated_weights_layer_2)[:self.weights_needed_for_layer_2]
            generated_weights = torch.cat([generated_weights_layer_1,generated_weights_layer_2])
        else:
            generated_weights = generated_weights_layer_1

        return generated_weights


class SinusoidHyperGenerator(torch.nn.Module):
    def __init__(self,config,device=None):
        super(SinusoidHyperGenerator, self).__init__()

        self.device = device
        if self.device is None:
           self.device = torch.device("cpu")

        self.hypernetwork = HyperNetworkSimple(z_dim = config["LAYER_1_EMBEDDING_DIM"],
                                               out_size = config["LAYER_1_OUT_DIM"],
                                               unit_weight_vec_size = 40)
        embedding_needed = 40 / config["LAYER_1_OUT_DIM"] 
        if embedding_needed.is_integer() is False:
            raise "embedding_needed is not int"
        embedding_needed = int(embedding_needed)
        self.embeddings = torch.nn.Parameter(torch.randn(embedding_needed,config["LAYER_1_EMBEDDING_DIM"]))
        #for i in range(embedding_needed):
        #    self.embeddings.append(torch.nn.Parameter(torch.randn(config["LAYER_1_EMBEDDING_DIM"])))

    def forward(self,custom_params=None):
        # here we know that the simple hypernetwork is able to take batches, no need for looping
        #all_embeddings = torch.stack(self.embeddings)
        if custom_params is None:
            return self.hypernetwork(self.embeddings,custom_params).view(-1)
        else:
            return self.hypernetwork(custom_params["generator_net.embeddings"],
                                     custom_params=custom_params,
                                     prefix="generator_net.hypernetwork").view(-1)


#######################
# HYPERNETWORK MODLES #
#######################


class HyperNetworkSimple(torch.nn.Module):
    def __init__(self,z_dim,out_size,unit_weight_vec_size,device=None):
        super(HyperNetworkSimple, self).__init__()
        
        self.device = device
        if self.device is None:
           self.device = torch.device("cpu")

        # The simple version is only using a single layer to project the embedding sapce into parameter space
        self.z_dim = z_dim
        self.unit_dim = unit_weight_vec_size
        self.out_size = out_size
        
        # owerwrite the initialization with our custom one, so instead of the layer having 1 gain,
        # we make it so the generated layer have a gain of 1. See the derivation for this in HyperNetworkSemiTwoLayer
        self.weight_init_std = np.sqrt(1 / (self.unit_dim * self.z_dim))
        self.bound = np.sqrt(3) * self.weight_init_std

        # self.layer = torch.nn.Linear(self.z_dim, self.out_size*self.unit_dim)

        self.weight = torch.nn.Parameter(torch.ones((self.out_size*self.unit_dim,self.z_dim)))
        with torch.no_grad():
            self.weight.uniform_(-self.bound,self.bound)
   
        self.bias = torch.nn.Parameter(torch.zeros(self.out_size*self.unit_dim)) # init biases to zero

        
    def forward(self, z,custom_params=None,prefix=None):
        #batch_size = z.shape[0]
        # return self.layer(z)#.view(batch_size,self.out_size,self.unit_dim)
        if custom_params is None:
            return torch.nn.functional.linear(z, self.weight, self.bias)
        else:
            return torch.nn.functional.linear(z, custom_params[prefix+".weight"], custom_params[prefix+".bias"])
        

class HyperNetworkSemiTwoLayer(torch.nn.Module):

    def __init__(self,z_dim,in_size,out_size,unit_weight_vec_size,device=None):
        super(HyperNetworkSemiTwoLayer, self).__init__()

        self.device = device
        if self.device is None:
           self.device = torch.device("cpu")

        # the hypernet is generaing weights from embeddings
        # It generates the multiple of unit_weight_vec_size weight.
        # The output dim will be in_size * out_size * unit_weight_vec_size

        self.z_dim = z_dim
        self.unit_dim = unit_weight_vec_size
        self.out_size = out_size
        self.in_size = in_size

        # about initialization: pytorch initializes both weights biases with:
        # Uniform{-sqrt(k),sqrt(k)}  where k = 1 / in_features

        # What we want to do actually is to initialize the hypernetwork in a way, that it generates wigths which have 1 gain
        # General equation: Var(a[l]) = n[l-1] * Var(W[l]) * Var(a[l-1])  https://www.deeplearning.ai/ai-notes/initialization/
        # In this case we want the ratio of Var(a[l]) / Var(a[l-1]) to be 1 / fan_in_of_generated_net
        # If this is the case, the generated weight will have the variance of 1 / fan_in_of_generated_net, which is waht we want so it has a gain of 1.
        # Var(a[l]) / Var(a[l-1]) := 1 / fan_in_of_generated_net
        # 1 / fan_in_of_generated_net = n_embedding * Var(W[l])
        # Var(W[l]) = 1 / (fan_in_of_generated_net * n_embedding)  and that is how we need to initailize our hyper net

        # Now we have this extra complexity of using the semi 2 layer network with the matmul
        # What is the computation of a single weight:
        # Layer 1:  32 -> 1
        # Layer 2:  32 -> 1
        # The gain of the 2 togeather should be 1 / fan_in_of_generated_net
        # Var(a[l]) = n[l-1] * Var(W[l]) * Var(a[l-1])
        # Var(a[l-1]) = n[l-2] * Var(W[l-1]) * Var(a[l-2]) 
        # now substitue the second into the first
        # Var(a[l]) = n[l-1] * Var(W[l]) * n[l-2] * Var(W[l-1]) * Var(a[l-2]) 
        # Var(a[l]) / Var(a[l-2]) := 1 / fan_in_of_generated_net
        # 1 / fan_in_of_generated_net = n[l-1] * Var(W[l]) * n[l-2] * Var(W[l-1])
        # In hypernetwork variables:
        # 1 / fan_in_of_generated_net = n_embedding  * Var(W[l]) * n_embedding * Var(W[l-1])
        # Extra constraint: let Var(W[l]) equal Var(W[l-1])
        # 1 / fan_in_of_generated_net = n_embedding * Var(W) * n_embedding * Var(W)
        # Var(W) * Var(W) = 1 / (fan_in_of_generated_net * n_embedding * n_embedding)
        # Var(W) = sqrt(1 / (fan_in_of_generated_net * n_embedding * n_embedding))
        # From the variance the std is sqrt(var(W))

        # You can initialize your weight with normal or uniform
        # For uiform pytorch uses: bound = math.sqrt(3.0) * std,  (I don't how that is derived, but let us do the same)
        # For normal, just use the std

        # Also there is the gain of the activation function. For identity it is 1, for relu it is sqrt(2), 
        # it does not really matter much, especially since we only generate 1 or 2 layers, i am going to ignore it

        # here the fan_in_of_generated_net is self.unit_dim, this is not necessarily the case but it is for us now since 
        # one embedding generates the weights for a single neuron
        self.weight_init_std = np.sqrt(np.sqrt(1 / (self.unit_dim * self.z_dim * self.z_dim)))
        self.bound = np.sqrt(3) * self.weight_init_std


        # the github implelemntation multiplied with w2 first then w1, here i reversed it to the normal order 
        # explanation for the wierd init above!
        self.w1 = torch.nn.Parameter(torch.randn((self.z_dim, self.in_size*self.z_dim)))
        self.b1 = torch.nn.Parameter(torch.zeros(self.in_size*self.z_dim))

        # both w2 and b2 are reused multiple times (with the matmul, the input have multiple rows...)
        # this 2 layer structure is equivalent to a huge one layer network, but we reused a lot of weights... 
        self.w2 = torch.nn.Parameter(torch.randn((self.z_dim, self.out_size*self.unit_dim)))
        self.b2 = torch.nn.Parameter(torch.zeros((self.out_size*self.unit_dim)))

        with torch.no_grad():
            self.w1.uniform_(-self.bound,self.bound)
            self.w2.uniform_(-self.bound,self.bound)

    def forward(self, z,custom_params=None,prefix=None):
        
        if custom_params is None:
            h_in = torch.matmul(z, self.w1) + self.b1
            h_in = h_in.view(self.in_size, self.z_dim)

            h_final = torch.matmul(h_in, self.w2) + self.b2
            generated_weights = h_final.view(self.out_size, self.in_size, self.unit_dim)
        else:
            h_in = torch.matmul(z, custom_params[prefix+".w1"]) + custom_params[prefix+".b1"]
            h_in = h_in.view(self.in_size, self.z_dim)

            h_final = torch.matmul(h_in, custom_params[prefix+".w2"]) + custom_params[prefix+".b2"]
            generated_weights = h_final.view(self.out_size, self.in_size, self.unit_dim)

        return generated_weights







def create_policy(config,device=None):
    if config["POLICY_TYPE"] == "DIRECT":
        if config["TASK"] == "SINUSOID_REGRESSION":
            return FunctionalFullyConnectedNet(num_inputs=1,hidden_dims=config["FULLY_CONNECTED_HIDDEN_DIMS_SINUSOID"],num_outputs=1,config=config,device=device)
        elif config["TASK"] == "OMNIGLOT":
            return FunctionalFullyConnectedNet(num_inputs=784,hidden_dims=config["FULLY_CONNECTED_HIDDEN_DIMS_OMNIGLOT"],num_outputs=config["N_WAY"],config=config,device=device)

    elif config["POLICY_TYPE"] == "HYPER_SIMPLE" or config["POLICY_TYPE"] == "HYPER_2_LAYER" or config["POLICY_TYPE"] == "DPPN":
            return IndirectNet(config,device=device)
    else:
         raise "Unknown policy type!!"