from functools import partial
from random import random
from typing import Optional
from opendelta.utils.signature import get_arg_names_inside_func
from opendelta.utils.utils import *
from opendelta.utils.cuda import get_device
from opendelta.basemodel import DeltaBase
import loralib as lora
import torch.nn as nn
import torch
import math
from opendelta.delta_models.layers.activations import Activations
import inspect
from opendelta import BaseDeltaConfig
  


class AdapterLayer(nn.Module):
    r"""A layer of adapter tuning module. 
    """
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
        
        self.layer_id = AdapterLayer.get_layer_count()
        AdapterLayer.count_layer()
        
    
    def instantiate(self, hidden_dim):
        self.modulelist = nn.Sequential()
        self.modulelist.add_module("down_proj",nn.Linear(hidden_dim, self.bottleneck_dim, device=self.device))

        # select non-linearity
        self.modulelist.add_module("non_linear", Activations(self.non_linearity.lower()))

        self.modulelist.add_module("up_proj", nn.Linear(self.bottleneck_dim, self.hidden_dim,  device=self.device))

        # TODO:
        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        # if self.add_layer_norm_after:
        #     self.adapter_norm_after = nn.LayerNorm(self.input_size)

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
            # print(f"Got hidden dim hidden_dim {self.hidden_dim}")
            self.instantiate(hidden_dim=self.hidden_dim)


        adapter_output = self.modulelist(hiddens)
        modified_output = adapter_output + hiddens # TODO option: disable residual_connection
        if isinstance(output, tuple):
            output = (modified_output,) + output[1:]
        elif isinstance(output, torch.Tensor):
            output = modified_output
        else:
            raise TypeError
        return output
    
  

class AdapterConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a [`LoraModel`]

    """
    def __init__(
        self, 
        bottleneck_dim: Optional[int]=24, 
        non_linearity: Optional[str]='gelu_new',
        sequential: Optional[str] = True,
        **kwargs
    ): 
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])



class AdapterModel(DeltaBase):

    config_class = AdapterConfig
    delta_type = "adapter"
    default_modified_modules = ["attn", "ff"]
    def __init__(self,
                 backbone_model: nn.Module, 
                 bottleneck_dim: Optional[int]=24, 
                 non_linearity: Optional[str]='gelu_new',
                 sequential: Optional[str] = True,
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
        self.insert_sequential_module(ref, pre_caller=None, post_caller=adapterlayer.forward, delta_module=adapterlayer, name="adapter")
    
    def new_module_like(self, module):
        module_device = get_device(module)
        adapterlayer = AdapterLayer(bottleneck_dim=self.bottleneck_dim, non_linearity=self.non_linearity, device=module_device)
        self.delta_modules.append(adapterlayer)  
        return adapterlayer
    