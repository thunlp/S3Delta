
from opendelta.basemodel import DeltaBase
from opendelta.delta_configs import BaseDeltaConfig
from opendelta.delta_models.layers.low_rank_linear import LowRankLinear
from opendelta.delta_models.layers.activations import Activations
from typing import Optional
from opendelta.utils.signature import get_arg_names_inside_func
import torch.nn as nn
import torch
from functools import partial
from typing import Optional
from opendelta.utils.utils import *
from opendelta.utils.cuda import get_device
from opendelta.basemodel import DeltaBase
import loralib as lora
import torch.nn as nn
import torch
import math



class LowRankAdapterConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a [`LoraModel`]

    """
    def __init__(
        self, 
        reduction_factor=32,
        non_linearity="gelu_new",
        low_rank_w_init="glorot-uniform",
        low_rank_rank=1,
        **kwargs
    ): 
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])



class LowRankAdapter(nn.Module):
    """This is the low-rank adapter, in which each adapter is composed of two rank-one matrices.
    """
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
        r""" Get the hidden_states from the PLM's layer output, pass it into the low-rank adapter, 
        then combined with the main hidden_states. Finally pass it into the subsequent layer.

        """

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
            

        z = self.down_sampler(hiddens)
        z = self.activation(z)
        adapter_output = self.up_sampler(z)

        modified_output = adapter_output + hiddens # residual_connection
        if isinstance(output, tuple):
            output = (modified_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = modified_output
        else:
            raise TypeError
        return output






class LowRankAdapterModel(DeltaBase, nn.Module):

    config_class = LowRankAdapterConfig
    delta_type = "lowrankadapter"
    default_modified_modules = ['attn', 'ff']
    def __init__(self,
                 backbone_model: nn.Module, 
                 reduction_factor = 32,
                 non_linearity = "gelu_new",
                 low_rank_w_init = "glorot-uniform", 
                 low_rank_rank = 1,
                 modified_modules: Optional[bool] = None,
                 unfrozen_modules: Optional[bool] = None,
                 common_structure: Optional[bool] = None,
                 ):
        DeltaBase.__init__(self, 
                           backbone_model, 
                           modified_modules=modified_modules,
                           unfrozen_modules=unfrozen_modules,
                           common_structure=common_structure,
                           )
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # not registered in parent class
                setattr(self, arg_name, locals()[arg_name])

        self.delta_modules = nn.ModuleList()

        self.add_all_delta_to_backbone(self.backbone_model,
                                   self.modified_modules,
                                   )
    
        
    def add_all_delta_to_backbone(self, 
                                module: nn.Module, 
                                modified_modules: List[str],
                                ) -> nn.Module:
        for key, _ in module.named_modules():
            if self.find_key(key, modified_modules):
                self.update_module(module, key)
        if (not hasattr(module,'pseudo')) or module.pseudo == True:
            self._pseudo_data_to_instantiate(module)
        self.mark_as_delta()
        return module
    
    def update_module(self, module: nn.Module, key: str):
        _, _, ref = self.find_module(module, key)
        adapterlayer = self.new_module_like(ref)
        self.insert_sequential_module(ref, pre_caller=None, post_caller=adapterlayer.forward, delta_module=adapterlayer, name="low_rank_adapter")
    
    def new_module_like(self, module):
        module_device = get_device(module)
        adapterlayer = LowRankAdapter(reduction_factor = self.reduction_factor,
                                      non_linearity = self.non_linearity,
                                      low_rank_w_init = self.low_rank_w_init, 
                                      low_rank_rank = self.low_rank_rank,
                                      device=module_device)
        self.delta_modules.append(adapterlayer)  
        return adapterlayer
    