
import copy
from argparse import ArgumentDefaultsHelpFormatter
from hashlib import new
from typing import List, Optional, Union
from attr import has
import torch.nn.functional as F
import loralib as lora
import torch
import torch.nn as nn
from opendelta.basemodel import DeltaBase
from opendelta.delta_models.adapter import AdapterLayer, AdapterModel, AdapterConfig
from opendelta.delta_models.bitfit import BitFitModel, BitFitConfig, BiasLayer
from opendelta.delta_models.compacter import (CompacterModel,
                                              HyperComplexAdapterLayer, CompacterConfig)
from opendelta.delta_models.lora import LoraModel, LoraConfig
from opendelta.delta_models.low_rank_adapter import (LowRankAdapter,
                                                     LowRankAdapterModel, LowRankAdapterConfig)
from opendelta.delta_models.prefix import PrefixLayerT5, PrefixModel, PrefixConfig
from opendelta.utils.cuda import get_device
from opendelta.utils.structure_mapping import transform
from opendelta.utils.utils import *
from opendelta.utils.visualization import Visualization
from transformers.models.t5.modeling_t5 import T5Attention
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
from opendelta.utils.structure_mapping import CommonStructureMap
from opendelta.utils.signature import get_arg_names
from opendelta.utils.cuda import get_device


def print_alphas(model):
    ret = '='*25+'alphas'+'='*25+'\n'
    for n, p in model.named_parameters():
        if n.find('alpha') != -1:
            if hasattr(p, 'delta_name_info'):
                name_info = p.delta_name_info
            else:
                name_info = None
            ret += f'| {n}\t| {F.softmax(p.data,dim=-1)}|{name_info}|argmax:{torch.argmax(p.data,dim=-1)}\n'
    ret += '='*56+'\n'
    return ret


def is_leaf_module(module):
    r"""Whether the module is a leaf module
    """
    return len([n for n, _ in module.named_children()]) == 0


def non_module_param(module: nn.Module):
    module_names = [n for n, _ in module.named_modules()]
    ret = []
    for n, p in module.named_parameters():
        if not is_child_key(n, module_names):
            ret.append((n, p))
    return ret


def named_weights(model: nn.Module):
    res = []
    for n, p in model.named_parameters():
        if n.find('alphas') == -1 and p.requires_grad == True:
            res.append((n, p))
        else:
            continue
    return res


def weights(model: nn.Module):
    res = []
    for n, p in model.named_parameters():
        if n.find('alphas') == -1 and p.requires_grad == True:
            res.append(p)
        else:
            continue
    return res


def named_alphas(model: nn.Module):
    res = []
    for n, p in model.named_parameters():
        if n.find('alphas') != -1 and p.requires_grad == True:
            res.append((n, p))
        else:
            continue
    return res


def alphas(model: nn.Module):
    res = []
    for n, p in model.named_parameters():
        if n.find('alphas') != -1 and p.requires_grad == True:
            res.append(p)
        else:
            continue
    return res

def get_softmax_alpha(alphas, using_gumbel=False, tau=1, hard = False,dim=-1):
    if using_gumbel:
        return F.gumbel_softmax(alphas,tau=tau,hard=hard,dim=dim)
    else: 
        return F.softmax(alphas,dim=dim)

def get_z(alpha,t=1,r=1,l=0):
    u = torch.rand(1).to(alpha.device)
    s = torch.sigmoid((torch.log(u)-torch.log(1-u)+alpha)/t)
    s_ = s*(r-l)+l
    z = torch.min(torch.Tensor([1]).to(s_.device),torch.max(torch.Tensor([0]).to(s_.device),s_))
    return z

class NoneConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a [`NoneModel`]

    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name):  # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])
from delta_layers import BitFitParallelLayer,LoraParallelLayer,LowRankAdapterSequentialLayer,HyperComplexAdapterSequentialLayer
class MixLayerParallel(nn.Module):
    # config_class = None
    # delta_type = "mix"
    # default_modified_modules = ["attn", "ff"]
    addable_delta_type = ['None', 'Adapter', 'BitFit',
                          'Compacter', 'Lora', 'LowRankAdapter']
    addable_adapter_like_delta_type = [
        'Adapter', 'Compacter', 'LowRankAdapter']
    type2class= {
        'Adapter': AdapterLayer,
        'Compacter': HyperComplexAdapterLayer,
        'LowRankAdapter': LowRankAdapterSequentialLayer,
        'BitFit':BitFitParallelLayer,
        'Lora':LoraParallelLayer
    }

    def __init__(self,
                 backbone_model: nn.Module,
                 _parallel_alphas=None,
                 _sequential_alphas=None,
                 delta_args=True,
                #  modified_modules: Optional[bool] = None,
                #  unfrozen_modules: Optional[bool] = None,
                 common_structure: Optional[bool] = None
                 ):
        nn.Module.__init__(self)

        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name):
                setattr(self, arg_name, locals()[arg_name])

        self.parallel_modules = nn.ModuleList()
        self.parallel_modules_types = []
        self.num_parallel_modules = 0
        self.sequential_modules = nn.ModuleList()
        self.sequential_modules_types = []
        self.num_sequential_modules = 0
        # self.delta_params = nn.ParameterList()
        # self.delta_modules = []

        # self.need_original_module_output = False
        if self._parallel_alphas != None:
            setattr(self._parallel_alphas, '_is_delta', True)
            setattr(self._parallel_alphas, 'delta_name_info', self.parallel_modules_types)
        if self._sequential_alphas != None:
            setattr(self._sequential_alphas, '_is_delta', True)
            setattr(self._sequential_alphas, 'delta_name_info', self.sequential_modules_types)

        self._parallel_gate_alphas = None
        self._sequential_gate_alphas = None
        
        


    def add_alphas(self, _sequential_alphas=False,_parallel_alphas=False,_parallel_gate_alphas=False,_sequential_gate_alphas=False):
        if _sequential_alphas is not False:
            self._sequential_alphas = _sequential_alphas
            setattr(self._sequential_alphas, '_is_delta', True)
            setattr(self._sequential_alphas, 'delta_name_info', self.sequential_modules_types)

        if _parallel_alphas is not False:
            self._parallel_alphas = _parallel_alphas
            setattr(self._parallel_alphas, '_is_delta', True)
            setattr(self._parallel_alphas, 'delta_name_info', self.parallel_modules_types)

        if _parallel_gate_alphas is not False:
            self._parallel_gate_alphas = _parallel_gate_alphas
            setattr(self._parallel_gate_alphas, '_is_delta', True)
            # setattr(self._gate_alphas, 'delta_name_info', self.sequential_modules_types)

        if _sequential_gate_alphas is not False:
            self._sequential_gate_alphas = _sequential_gate_alphas
            setattr(self._sequential_gate_alphas, '_is_delta', True)

        

    def add_delta_from_config(self, delta_type: str, insert_method: str, config: Union[BaseDeltaConfig, dict]):

        if delta_type==None:
            assert self.delta_args.using_gate==False, "if using gate, None is prohibited"
        # verify insert_method
        assert insert_method in ["parallel", "sequential"]

        # verify&set _add_{delta_type}=True
        assert delta_type in self.addable_delta_type, f'delta_type should be in {self.addable_delta_type}'
        # assert hasattr(
        #     self, f'_add_{delta_type}') == False, f'already insert {delta_type}'
        setattr(self, f'_add_{delta_type}', True)

        
        if delta_type == 'Lora':
            layertype = self.type2class[delta_type]
            if isinstance(self.backbone_model, nn.Linear):
                in_features, out_features = self.backbone_model.in_features, self.backbone_model.out_features
                new_module = layertype(in_features=in_features, 
                                        out_features=out_features, 
                                        r=config.lora_r, 
                                        lora_alpha=config.lora_alpha,
                                        lora_dropout=config.lora_dropout)
            else:
                raise NotImplementedError
        elif delta_type == 'BitFit':
            layertype = self.type2class[delta_type]
            if isinstance(self.backbone_model, nn.Linear):
                in_features, out_features = self.backbone_model.in_features, self.backbone_model.out_features
                new_module = layertype(out_features=out_features)
            else:
                raise NotImplementedError

        elif delta_type != 'None':
            layertype = self.type2class[delta_type]

            arg_names = get_arg_names(layertype)
            args = {}
            if 'device' in arg_names:
                args['device'] = get_device(self.backbone_model)
            for arg_name in arg_names:
                if arg_name in config.__dict__:
                    args[arg_name] = getattr(config, arg_name)

            new_module = layertype(**args)
        else: #delta_type == 'None':
            new_module = None

        modules_list = getattr(self,f"{insert_method}_modules")
        modules_list.append(new_module)
        modules_type_list = getattr(self,f"{insert_method}_modules_types")
        modules_type_list.append(delta_type)
        setattr(self,f"num_{insert_method}_modules",getattr(self,f"num_{insert_method}_modules")+1)


    def forward(self, *args, **kwargs):

        if self._parallel_alphas == None and self.num_parallel_modules>0:
            alphas = nn.Parameter(torch.ones(
                self.num_parallel_modules)/1e3, requires_grad=True)
            self.add_alphas(_parallel_alphas=alphas)
            # setattr(self._parallel_alphas, '_is_delta', True)
        if self._sequential_alphas == None and self.num_sequential_modules>0:
            alphas = nn.Parameter(torch.ones(
                self.num_sequential_modules)/1e3, requires_grad=True)
            self.add_alphas(_sequential_alphas=alphas)
        if self.delta_args.using_gate and self._parallel_gate_alphas==None and self.num_parallel_modules>0:
            alphas = nn.Parameter(torch.ones(
                1)/1e3, requires_grad=True)
            self.add_alphas(_parallel_gate_alphas=alphas)
        if self.delta_args.using_gate and self._sequential_gate_alphas == None and self.num_sequential_modules>0:
            alphas = nn.Parameter(torch.ones(
                1)/1e3, requires_grad=True)
            self.add_alphas(_sequential_gate_alphas=alphas)


        original_output = self.backbone_model(*args, **kwargs)

        
        def add_hiddens(output):
            if isinstance(output, tuple):
                hiddens = output[0]
            elif isinstance(output, torch.Tensor):
                hiddens = output
            else:
                raise TypeError
            hidden_states_list.append(hiddens)
        def get_hiddens(output):
            if isinstance(output, tuple):
                hiddens = output[0]
            elif isinstance(output, torch.Tensor):
                hiddens = output
            else:
                raise TypeError
            return hiddens

        if self.num_parallel_modules>0:
            hidden_states_list = []
            for idx, parallel_module in enumerate(self.parallel_modules):
                if parallel_module is None:
                    output = torch.zeros_like(get_hiddens(original_output))
                else:
                    output = parallel_module(*args, **kwargs)
                add_hiddens(output)
            
            assert self._parallel_alphas.shape[0] == len(hidden_states_list)
            softmax_alphas = get_softmax_alpha(self._parallel_alphas,self.delta_args.using_gumbel,self.delta_args.t1,hard=self.delta_args.gumbel_hard)
            new_hidden_states = torch.sum(softmax_alphas * torch.stack(hidden_states_list, dim=-1), dim=-1)
            if self.delta_args.using_gate:
                z = get_z(self._parallel_gate_alphas,t=self.delta_args.t0,r=self.delta_args.r,l=self.delta_args.l)
                new_hidden_states = z*new_hidden_states
            if isinstance(original_output, tuple):
                parallel_output = (new_hidden_states+original_output[0],) + original_output[1:]
            elif isinstance(original_output, torch.Tensor):
                parallel_output = new_hidden_states+original_output
            else:
                raise TypeError
        else:
            parallel_output = original_output
        
        if len(self.sequential_modules) > 0:
            hidden_states_list = []
            for idx, sequential_module in enumerate(self.sequential_modules):
                if sequential_module == None:
                    output = torch.zeros_like(get_hiddens(parallel_output))
                else:
                    # print(type(self.parallel_modules[idx]))
                    output = sequential_module(parallel_output)
                add_hiddens(output)
            
            assert self._sequential_alphas.shape[0] == len(hidden_states_list)
            new_hidden_states = torch.sum(F.softmax(self._sequential_alphas, dim=-1) * torch.stack(hidden_states_list, dim=-1), dim=-1)
            if isinstance(parallel_output, tuple):
                sequential_output = (new_hidden_states,) + parallel_output[1:]
            elif isinstance(parallel_output, torch.Tensor):
                sequential_output = new_hidden_states
            else:
                raise TypeError
        else:
            sequential_output = parallel_output
        return sequential_output



        # if self.num_parallel_modules == 1:
        #     parallel_module = self.delta_modules[0][0]
        #     if parallel_module == 'None':
        #         output = self.backbone_model(*args, **kwargs)
        #     elif parallel_module in self.addable_adapter_like_delta_type:
        #         output = self.parallel_modules[0](
        #             self.backbone_model(*args, **kwargs))

        #     else:
        #         output = self.parallel_modules[0](*args, **kwargs)

        #     if isinstance(output, tuple):
        #         output = ((F.softmax(self._alphas, dim=-1)
        #                   * output[0]),) + output[1:]
        #     elif isinstance(output, torch.Tensor):
        #         output = F.softmax(self._alphas, dim=-1)*output
        #     else:
        #         raise TypeError
        #     return output

        # original_output = None
        # # original_output = self.backbone_model(*args, **kwargs)
        # hidden_states_list = []

        # def add_hiddens(output):
        #     if isinstance(output, tuple):
        #         hiddens = output[0]
        #     elif isinstance(output, torch.Tensor):
        #         hiddens = output
        #     else:
        #         raise TypeError
        #     hidden_states_list.append(hiddens)

        # if hasattr(self, '_add_None') and self._add_None == True:
        #     add_hiddens(original_output)
        # for idx, parallel_module in enumerate(self.delta_modules):
        #     if parallel_module[0] == 'None':
        #         if original_output == None:
        #             original_output = self.backbone_model(*args, **kwargs)
        #         output = original_output
        #     elif parallel_module[0] in self.addable_adapter_like_delta_type:
        #         if original_output == None:
        #             original_output = self.backbone_model(*args, **kwargs)
        #         output = self.parallel_modules[idx](original_output)

        #     else:
        #         # print(type(self.parallel_modules[idx]))
        #         output = self.parallel_modules[idx](*args, **kwargs)
        #     add_hiddens(output)

        # assert self._alphas.shape[0] == len(hidden_states_list)
        # new_hidden_states = torch.sum(
        #     F.softmax(self._alphas, dim=-1) * torch.stack(hidden_states_list, dim=-1), dim=-1)

        # if isinstance(original_output, tuple):
        #     output = (new_hidden_states,) + original_output[1:]
        # elif isinstance(original_output, torch.Tensor):
        #     output = new_hidden_states
        # else:
        #     raise TypeError

        # return output

    # def find_key(self, key: Union[str, re.Pattern], target_list: List[str], only_tail=True):
    #     if key == '' and target_list == ['']:
    #         return True
    #     if self.common_structure and not hasattr(self, 'structure_mapping'):

    #         self.structure_mapping = CommonStructureMap.load(
    #             self.backbone_model)
    #     if self.common_structure:
    #         key = self.structure_mapping.transform(key, strict=False)
    #     if not key:
    #         return False
    #     try:
    #         if isinstance(key, re.Pattern):  # TODO: unit test needed ERROR
    #             if only_tail:
    #                 return endswith_in_regex(key, target_list)
    #             else:
    #                 return substring_in_regex(key, target_list)
    #         else:
    #             if only_tail:
    #                 return endswith_in(key, target_list)
    #             else:
    #                 return substring_in(key, target_list)
    #     except:
    #         from IPython import embed
    #         embed(header="exception")

    # def find_module(self, root_module: nn.Module, key: str):
    #     if key == '':
    #         return root_module, 'new_module_for_replacing', root_module
    #     sub_keys = key.split(".")
    #     parent_module = root_module
    #     for sub_key in sub_keys[:-1]:
    #         parent_module = getattr(parent_module, sub_key)
    #     module = getattr(parent_module, sub_keys[-1])
    #     return parent_module, sub_keys[-1], module


class S3Delta(DeltaBase):
    default_modified_modules = [""]
    unfrozen_modules = [
        "deltas",
        "layer_norm",
        "final_layer_norm"
    ]

    def __init__(self,
                 backbone_model: nn.Module,
                    delta_args,
                 unfrozen_modules: Optional[List[str]] = None,
                 common_structure=False,
                 ):

        DeltaBase.__init__(self,
                           backbone_model,
                           modified_modules=[],
                           unfrozen_modules=unfrozen_modules,
                           common_structure=common_structure,)
        self.delta_args = delta_args
        self.new_module_list = []
        # backbone = self.backbone_model
        self.plm_total_params = sum(p.numel() for p in self.backbone_model.parameters())

    def modify_backbone_from_config(self,modified_configs: dict):
        self.modified_configs = modified_configs
        modified_modules = list(modified_configs.keys())
        for key, _ in self.backbone_model.named_modules():
            if self.find_key(key, modified_modules):
                print("find key", key)
                self.update_module(self.backbone_model, key)

        self._pseudo_data_to_instantiate(self.backbone_model)
        self.mark_as_delta()

        self.new_module_list = []

    def find_config(self, key):
        for i in self.modified_configs:
            if self.find_key(key, [i]):
                return self.modified_configs[i]

    def update_module(self, module: nn.Module, key: str):
        parent_ref, children_name, child_ref = self.find_module(module, key)
        modified_config = self.find_config(key)
        self.replace_module(parent_ref, children_name,
                            child_ref, modified_config)

    def replace_module(self, parent_module: nn.Module, children_name: str, replacedmodule: Optional[nn.Module] = None, modified_config=None):
        new_module = MixLayerParallel(
            replacedmodule, delta_args = self.delta_args,common_structure=self.common_structure)
        self.new_module_list.append(new_module)
        for insert_method in modified_config:
            for parallel_module_name in modified_config[insert_method]:
            # print(parallel_module_name)
                new_module.add_delta_from_config(parallel_module_name, insert_method,
                                             (globals().get(f'{parallel_module_name}Config'))(**(modified_config[insert_method][parallel_module_name])),)
        setattr(parent_module, children_name, new_module)
    def mark_as_delta(self, module: nn.Module = None,):
        if module==None:
            module=self
        for new_module in module.new_module_list:
            for parallel_module in new_module.parallel_modules:
                if parallel_module is not None:
                    for p in parallel_module.parameters():
                            setattr(p, "_is_delta", True)
            for sequential_module in new_module.sequential_modules:
                if sequential_module is not None:
                    for p in sequential_module.parameters():
                            setattr(p, "_is_delta", True)

    
    def num_backbone_parameters(self):
        return self.plm_total_params

    # def freeze_not_delta_parameter(self, module: nn.Module):

    #     for n, p in module.named_parameters():
    #         if hasattr(p, "_is_delta") and getattr(p, "_is_delta") == True:
    #             p.requires_grad = True
    #         else:
    #             p.requires_grad = False
    #     return

    # def named_trainable_parameters_without_alphas(self, model: nn.Module):
    #     for n, p in model.named_parameters():
    #         if n.find('alphas') == -1 and p.requires_grad == True:
    #             yield n, p
    #         else:
    #             continue

    # def trainable_parameters_without_alphas(self, model: nn.Module):
    #     for n, p in self.named_trainable_parameters_without_alphas(model):
    #         yield p

    # def named_trainable_alphas(self, model: nn.Module):
    #     for n, p in model.named_parameters():
    #         if n.find('alphas') != -1 and p.requires_grad == True:
    #             yield n, p
    #         else:
    #             continue
