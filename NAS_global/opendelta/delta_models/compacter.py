from functools import partial
from typing import Optional
from opendelta.delta_configs import BaseDeltaConfig
from opendelta.delta_models.adapter import AdapterLayer
from opendelta.utils.signature import get_arg_names_inside_func
from opendelta.utils.utils import *
from opendelta.utils.cuda import get_device
from opendelta.basemodel import DeltaBase
import loralib as lora
import torch.nn as nn
import torch
import math
import opendelta
from opendelta.delta_models.layers.activations import Activations
import inspect
from opendelta.delta_models.layers.hypercomplex_linear import PHMLinear


class HyperComplexAdapterLayer(nn.Module):
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
        r""" Get the hidden_states from the PLM's layer output, pass it into the hypercomplex adapter, 
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

class CompacterConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a [`LoraModel`]

    """
    def __init__(
        self, 
        bottleneck_dim: Optional[int]=32, 
        non_linearity: Optional[str]='relu',
        sequential: Optional[str] = True,
        reduction_factor=16, 
        phm_c_init="normal", 
        hypercomplex_division=4,
        learn_phm=True,
        hypercomplex_nonlinearity="glorot-uniform",
        shared_phm_rule=False,
        factorized_phm=True,
        shared_W_phm=False,
        factorized_phm_rule=False,
        phm_rank=1,
        phm_init_range=0.0001,
        kronecker_prod=None,
        use_bias_up_sampler=True,
        use_bias_down_sampler=True,
        **kwargs
    ): 
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])



class CompacterModel(DeltaBase, nn.Module):
    config_class = CompacterConfig
    delta_type = "compacter"
    default_modified_modules = ["attn", "ff"]
    def __init__(self, 
                 backbone_model,
                 modified_modules: Optional[bool] = None,
                 unfrozen_modules: Optional[bool] = None,
                 common_structure: Optional[bool] = None,
                 structure_mapping=None,
                 reduction_factor=16, 
                 non_linearity="gelu_new", 
                 phm_c_init="normal", 
                 hypercomplex_division=4,
                 learn_phm=True,
                 hypercomplex_nonlinearity="glorot-uniform",
                 shared_phm_rule=False,
                 factorized_phm=True,
                 shared_W_phm=False,
                 factorized_phm_rule=False,
                 phm_rank=1,
                 phm_init_range=0.0001,
                 kronecker_prod=None,
                 use_bias_up_sampler=True,
                 use_bias_down_sampler=True,
                ):
        DeltaBase.__init__(self, 
                           backbone_model, 
                           modified_modules=modified_modules,
                           unfrozen_modules=unfrozen_modules,
                           common_structure=common_structure,
                           )
        assert shared_phm_rule == False, "In opendelta version {opendelta.__version__}, "\
            "shared_phm_rule is not supported. Later, sharing parameters will be tackled using"\
            "a unified paradigm."
        assert shared_W_phm == False, "In opendelta version {opendelta.__version__}, "\
            "shared_W_phm is not supported. Later, sharing parameters will be tackled using"\
            "a unified paradigm."
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
        self.insert_sequential_module(ref, 
                                      pre_caller=None, 
                                      post_caller=adapterlayer.forward, 
                                      delta_module=adapterlayer,
                                      name="compactor")
    
    def new_module_like(self, module):
        module_device = get_device(module)
        adapterlayer = HyperComplexAdapterLayer(reduction_factor=self.reduction_factor, 
                                                non_linearity=self.non_linearity, 
                                                phm_c_init=self.phm_c_init, 
                                                hypercomplex_division=self.hypercomplex_division,
                                                learn_phm=self.learn_phm,
                                                hypercomplex_nonlinearity=self.hypercomplex_nonlinearity,
                                                shared_phm_rule=self.shared_phm_rule,
                                                factorized_phm=self.factorized_phm,
                                                shared_W_phm=self.shared_W_phm,
                                                factorized_phm_rule=self.factorized_phm_rule,
                                                phm_rank=self.phm_rank,
                                                phm_init_range=self.phm_init_range,
                                                kronecker_prod=self.kronecker_prod,
                                                use_bias_up_sampler=self.use_bias_up_sampler,
                                                use_bias_down_sampler=self.use_bias_down_sampler,
                                                device=module_device
                                                )
        self.delta_modules.append(adapterlayer)  
        return adapterlayer
    