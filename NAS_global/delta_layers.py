import math
from typing import Optional

import loralib as lora
import torch
import torch.nn as nn

from opendelta import BaseDeltaConfig
from opendelta.delta_models.layers.activations import Activations
from opendelta.delta_models.layers.hypercomplex_linear import PHMLinear
from opendelta.delta_models.layers.low_rank_linear import LowRankLinear

@torch.no_grad()
def reset_param(module:nn.Module):
    if isinstance(module,T5LayerNormParalleyLayer):
        module.weight.zero_()
    elif isinstance(module,BitFitParallelLayer):
        module.bias.zero_()
    elif isinstance(module,LoraParallelLayer):
        module.reset_parameters()
    elif isinstance(module,LowRankAdapterSequentialLayer):
        module.down_sampler.reset_parameters()
        module.up_sampler.reset_parameters()
    else:
        raise NotImplementedError

class NoneConfig(BaseDeltaConfig):
    def __init__(self):
        super().__init__()
class LNFitConfig(BaseDeltaConfig):
    def __init__(self):
        super().__init__()
class BitFitParallelConfig(BaseDeltaConfig):
    def __init__(self):
        super().__init__()
class BitFitSequentialConfig(BaseDeltaConfig):
    def __init__(self):
        super().__init__()
class T5LayerNormParalleyLayer(nn.Module):
    def __init__(self, eps=1e-6,init_method="zero"):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.variance_epsilon = eps
        self.instantiated = False
        self.init_method=init_method
    def instantiate(self, hidden_dim):
        if self.init_method == "zero":
            self.weight = nn.Parameter(torch.zeros(hidden_dim))
        else:
            raise NotImplementedError
        self.instantiated = True
    def forward(self, hidden_states):

        if not self.instantiated:
            self.hidden_dim = hidden_states.shape[-1]
            self.instantiate(hidden_dim=self.hidden_dim)
            
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

class BitFitParallelLayer(nn.Module):
    def __init__(self, init_method="zero"):
        super().__init__()
        self.init_method=init_method
        self.instantiated = False
        
    def instantiate(self, hidden_dim):
        if self.init_method == "zero":
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
        else:
            raise NotImplementedError
        self.instantiated = True
    def forward(self, output):
        if isinstance(output, tuple):
            hiddens = output[0]
        elif isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError
        
        if not self.instantiated:
            self.hidden_dim = hiddens.shape[-1]
            # print(f"Got hidden dim hidden_dim {self.hidden_dim}")
            self.instantiate(hidden_dim=self.hidden_dim)

        modified_output =  torch.zeros_like(hiddens) + self.bias
        
        if isinstance(output, tuple):
            output = (modified_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = modified_output
        else:
            raise TypeError
        return output
class BitFitSequentialLayer(nn.Module):
    def __init__(self, init_method="zero"):
        super().__init__()
        self.init_method=init_method
        self.instantiated = False

    def instantiate(self, hidden_dim):
        if self.init_method == "zero":
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
        else:
            raise NotImplementedError
        self.instantiated = True
    
    def forward(self, output):
        if isinstance(output, tuple):
            hiddens = output[0]
        elif isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError
        
        if not self.instantiated:
            self.hidden_dim = hiddens.shape[-1]
            self.instantiate(hidden_dim=self.hidden_dim)

        modified_output = hiddens + self.bias
        
        if isinstance(output, tuple):
            output = (modified_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = modified_output
        else:
            raise TypeError
        return output    
class LoraParallelLayer(nn.Module, lora.LoRALayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Module.__init__(self)
        lora.LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            # self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            # self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            # self.weight.requires_grad = False
        self.reset_parameters()
        # if fan_in_fan_out:
        #     self.weight.data = self.weight.data.T

    def reset_parameters(self):
        # nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return x

class AdapterSequentialLayer(nn.Module):
    layer_count = 0

    @classmethod
    def count_layer(cls):
        cls.layer_count += 1
    
    @classmethod
    def get_layer_count(cls):
        return cls.layer_count

    def __init__(self, bottleneck_dim=24, non_linearity='gelu_new', device=None):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.device = device
        self.instantiated = False
        self.non_linearity = non_linearity
        
        self.layer_id = AdapterSequentialLayer.get_layer_count()
        AdapterSequentialLayer.count_layer()
        
    
    def instantiate(self, hidden_dim):
        self.modulelist = nn.Sequential()
        self.modulelist.add_module("down_proj",nn.Linear(hidden_dim, self.bottleneck_dim))#device=self.device

        # select non-linearity
        self.modulelist.add_module("non_linear", Activations(self.non_linearity.lower()))

        self.modulelist.add_module("up_proj", nn.Linear(self.bottleneck_dim, self.hidden_dim))#device=self.device

        self.instantiated = True
        # initialize the weight, which is important for fast convergence and better performance. 
        self.apply(self._init_weight)
    
    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01) 
            if module.bias is not None:
                module.bias.data.zero_()
        
    
    def forward(self, output):
        r""" Get the hidden_states from the PLM's layer output, pass it into the adapter, 
        then combined with the main hidden_states. Finally pass it into the subsequent layer.

        """
        # if self.instantiated and self.layer_id == 0 and random() < 0.01:
        #     print(self.modulelist[0].weight)
        if isinstance(output, tuple):
            hiddens = output[0]
        elif isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError
        
        if not self.instantiated:
            self.hidden_dim = hiddens.shape[-1]
            self.instantiate(hidden_dim=self.hidden_dim)


        adapter_output = self.modulelist(hiddens)
        modified_output = adapter_output
        if isinstance(output, tuple):
            output = (modified_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = modified_output
        else:
            raise TypeError
        return output


class HyperComplexAdapterSequentialLayer(nn.Module):
    """Hypercomplex Adapter layer, in which the weights of up and down sampler modules
    are parameters are 1/n times of the conventional adapter layers, where n is
    hypercomplex division number."""

    def __init__(self, 
                 reduction_factor=16, 
                 non_linearity="relu", 
                 phm_c_init="normal", 
                 hypercomplex_division=4,
                 learn_phm=True,
                 hypercomplex_nonlinearity="glorot-uniform",
                 shared_phm_rule=False,
                 factorized_phm=True,
                 phm_rule: Optional[torch.Tensor]=None,
                 shared_W_phm=False,
                 factorized_phm_rule=False,
                 phm_rank=1,
                 phm_init_range=0.0001,
                 kronecker_prod=None,
                 device=None,
                 use_bias_up_sampler=True,
                 use_bias_down_sampler=True,
                 ):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.non_linearity = non_linearity
        self.phm_c_init = phm_c_init
        self.hypercomplex_division = hypercomplex_division
        self.learn_phm = learn_phm
        self.phm_rule=phm_rule
        self.hypercomplex_nonlinearity = hypercomplex_nonlinearity
        self.shared_phm_rule = shared_phm_rule
        self.factorized_phm = factorized_phm
        self.shared_W_phm = shared_W_phm
        self.factorized_phm_rule = factorized_phm_rule
        self.phm_rank = phm_rank
        self.phm_init_range = phm_init_range
        self.kronecker_prod = kronecker_prod
        self.use_bias_up_sampler=use_bias_up_sampler
        self.use_bias_down_sampler=use_bias_down_sampler
        self.device = device

        self.instantiated = False
        
    def instantiate(self, hidden_dim):
        self.down_sample_size = hidden_dim // self.reduction_factor
        self.activation = Activations(self.non_linearity.lower()).to(self.device)
        self.down_sampler = PHMLinear(in_features=hidden_dim,
                                      out_features=self.down_sample_size,
                                      bias=self.use_bias_down_sampler,
                                      c_init=self.phm_c_init,
                                      phm_dim=self.hypercomplex_division,
                                      phm_rule=self.phm_rule,
                                      learn_phm=self.learn_phm,
                                      w_init=self.hypercomplex_nonlinearity,
                                      shared_phm_rule=self.shared_phm_rule,
                                      factorized_phm=self.factorized_phm,
                                      shared_W_phm=self.shared_W_phm,
                                      factorized_phm_rule=self.factorized_phm_rule,
                                      phm_rank=self.phm_rank,
                                      phm_init_range=self.phm_init_range,
                                      kronecker_prod=self.kronecker_prod).to(self.device)
        self.up_sampler = PHMLinear(in_features=self.down_sample_size,
                                    out_features=hidden_dim, 
                                    bias=self.use_bias_up_sampler,
                                    c_init=self.phm_c_init,
                                    phm_dim=self.hypercomplex_division,
                                    phm_rule=self.phm_rule,
                                    learn_phm=self.learn_phm,
                                    w_init=self.hypercomplex_nonlinearity,
                                    shared_phm_rule=self.shared_phm_rule,
                                    factorized_phm=self.factorized_phm,
                                    shared_W_phm=self.shared_W_phm,
                                    factorized_phm_rule=self.factorized_phm_rule,
                                    phm_rank=self.phm_rank,
                                    phm_init_range=self.phm_init_range,
                                    kronecker_prod=self.kronecker_prod).to(self.device)
        self.instantiated = True

    def forward(self, output):
        if isinstance(output, tuple):
            hiddens = output[0]
        elif isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError
        
        if not self.instantiated:
            self.hidden_dim = hiddens.shape[-1]
            self.instantiate(hidden_dim=self.hidden_dim)
            

        z = self.down_sampler(hiddens)
        z = self.activation(z)
        adapter_output = self.up_sampler(z)

        modified_output = adapter_output
        if isinstance(output, tuple):
            output = (modified_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = modified_output
        else:
            raise TypeError
        return output


class LowRankAdapterSequentialLayer(nn.Module):
    def __init__(self, 
                 reduction_factor=32, 
                 non_linearity="gelu_new",
                 low_rank_w_init="glorot-uniform", 
                 low_rank_rank=1,
                 device=None):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.non_linearity = non_linearity
        self.low_rank_w_init = low_rank_w_init
        self.low_rank_rank = low_rank_rank
        self.device = device
        self.instantiated = False

    def instantiate(self, hidden_dim):
        self.down_sample_size = hidden_dim // self.reduction_factor
        self.activation = Activations(self.non_linearity.lower()).to(self.device)
        self.down_sampler = LowRankLinear(hidden_dim, self.down_sample_size,
                                          w_init=self.low_rank_w_init,
                                          rank=self.low_rank_rank).to(self.device)
        self.up_sampler = LowRankLinear(self.down_sample_size, hidden_dim,
                                        w_init=self.low_rank_w_init,
                                        rank=self.low_rank_rank).to(self.device)

        self.instantiated = True

    def forward(self, output):
        if isinstance(output, tuple):
            hiddens = output[0]
        elif isinstance(output, torch.Tensor):
            hiddens = output
        else:
            raise TypeError
        
        if not self.instantiated:
            self.hidden_dim = hiddens.shape[-1]
            self.instantiate(hidden_dim=self.hidden_dim)
            
        z = self.down_sampler(hiddens)
        z = self.activation(z)
        adapter_output = self.up_sampler(z)

        modified_output = adapter_output
        if isinstance(output, tuple):
            output = (modified_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = modified_output
        else:
            raise TypeError
        return output
