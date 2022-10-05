
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
from opendelta.delta_models.bitfit import BitFitModel, BitFitConfig
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
    ret = ''
    ret += '='*25+'alphas'+'='*25+'\n'
    # print('='*25+'alphas'+'='*25)
    for n, p in model.named_parameters():
        if n.find('alpha') != -1:
            # print(
            # f'| {n}\t| {F.softmax(p.data,dim=-1)}| argmax:{torch.argmax(p.data,dim=-1)}')
            if hasattr(p, 'delta_name_info'):
                name_info = p.delta_name_info
            else:
                name_info = None
            ret += f'| {n}\t| {F.softmax(p.data,dim=-1)}|{name_info}|argmax:{torch.argmax(p.data,dim=-1)}\n'
    # print('='*56)
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


class MixLayerParallel(nn.Module):
    config_class = None
    delta_type = "mix"
    default_modified_modules = ["attn", "ff"]
    addable_delta_type = ['None', 'Adapter', 'BitFit',
                          'Compacter', 'Lora', 'LowRankAdapter', 'Prefix']
    addable_adapter_like_delta_type = [
        'Adapter', 'Compacter', 'LowRankAdapter']
    adapter_like_layer_mapping = {
        'Adapter':AdapterLayer,
        'Compacter':HyperComplexAdapterLayer,
        'LowRankAdapter':LowRankAdapter
    }

    def __init__(self,
                 backbone_model: nn.Module,
                 _alphas=None,
                 modified_modules: Optional[bool] = None,
                 unfrozen_modules: Optional[bool] = None,
                 common_structure: Optional[bool] = None
                 ):
        nn.Module.__init__(self)

        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name):  
                setattr(self, arg_name, locals()[arg_name])

        self.parallel_modules = nn.ModuleList()
        # self.delta_params = nn.ParameterList()
        self.delta_modules = []

        # self.need_original_module_output = False
        self.num_parallel_module = 0

    def copy_module(self, module: nn.Module, for_bias=False, modified_keys=None):
        '''share parameter'''

        if for_bias:
            assert modified_keys != None

        new_module = copy.deepcopy(module)
        for n, p in new_module.named_parameters():
            org_param = module.get_parameter(n)
            p.data = org_param.data
            # print(n)
            if hasattr(org_param,'_is_delta'):
                if not hasattr(new_module.get_parameter(n),'_is_delta'):
                    print(f"not have delta: {n}")
                    setattr(p,'_is_delta',getattr(org_param,'_is_delta'))
                else: 
                    print(f"have delta: {n}")
            #     p.requires_grad = True

        if for_bias:
            for key, _ in new_module.named_modules():
                if self.find_key(key, modified_keys):
                    # print("find key",key)
                    _, _, ref = self.find_module(module, key)
                    for n, c in ref.named_modules():
                        if isinstance(c, nn.Linear):
                            if c.bias is None:
                                pass
                            else:
                                bias = copy.deepcopy(c.bias)
                                c.register_parameter('bias', bias)
                                # _reset_bias_parameters(c)
                                c.bias.requires_grad = True
                        else:
                            pass

        return new_module

    def add_alphas(self, alphas):
        self._alphas = alphas
        setattr(self._alphas, '_is_delta', True)

    def add_delta_from_config(self, delta_type: str, config: Union[BaseDeltaConfig, dict]):

        assert delta_type in self.addable_delta_type, f'delta_type should be in {self.addable_delta_type}'
        assert hasattr(
            self, f'_add_{delta_type}') == False, f'already insert {delta_type}'
        setattr(self, f'_add_{delta_type}', True)

        if delta_type == 'BitFit':
            for_bias = True
        else:
            for_bias = False

        if delta_type in self.addable_adapter_like_delta_type and config.modified_modules==[""]:
            # do not need to copy module
            layertype = self.adapter_like_layer_mapping[delta_type]
            arg_names = get_arg_names(layertype)
            # print(arg_names)
            args = {}
            if 'device' in arg_names:
                args['device'] = get_device(self.backbone_model)
            for arg_name in arg_names:
                if arg_name in config.__dict__:
                    args[arg_name] = getattr(config, arg_name)
            new_layer = layertype(**args)
            self.delta_modules.append((delta_type,new_layer))
            self.parallel_modules.append(new_layer)

        elif delta_type != 'None':
            delta_class = globals().get(f'{delta_type}Model')
            parallel_module = self.copy_module(
                self.backbone_model, for_bias=for_bias, modified_keys=config.modified_modules)
            setattr(parallel_module, 'pseudo', False)
            delta_model = delta_class.from_config(config, parallel_module)
            if delta_type == 'Lora':
                if delta_model.modified_modules == ['']:
                    parallel_module = getattr(
                        parallel_module, 'new_module_for_replacing')
            self.delta_modules.append((delta_type,delta_model))
            self.parallel_modules.append(parallel_module)
        else:
            self.delta_modules.append((delta_type,None))
            self.parallel_modules.append(None)
        self.num_parallel_module += 1


    def forward(self, *args, **kwargs):

        if self._alphas == None:
            self._alphas = nn.Parameter(torch.ones(
                self.num_parallel_module)/1e3, requires_grad=True)
            setattr(self._alphas, '_is_delta', True)

        if self.num_parallel_module == 1:
            parallel_module = self.delta_modules[0][0]
            if parallel_module == 'None':
                output = self.backbone_model(*args, **kwargs)
            elif parallel_module in self.addable_adapter_like_delta_type:
                output = self.parallel_modules[0](self.backbone_model(*args, **kwargs))

            else:
                output = self.parallel_modules[0](*args, **kwargs)


            if isinstance(output, tuple):
                output = ((F.softmax(self._alphas, dim=-1)*output[0]),) + output[1:]
            elif isinstance(output, torch.Tensor):
                output = F.softmax(self._alphas, dim=-1)*output
            else:
                raise TypeError
            return output

        original_output = None
        # original_output = self.backbone_model(*args, **kwargs)
        hidden_states_list = []

        def add_hiddens(output):
            if isinstance(output, tuple):
                hiddens = output[0]
            elif isinstance(output, torch.Tensor):
                hiddens = output
            else:
                raise TypeError
            hidden_states_list.append(hiddens)

        # if hasattr(self, '_add_None') and self._add_None == True:
        #     add_hiddens(original_output)
        for idx, parallel_module in enumerate(self.delta_modules):
            if parallel_module[0] == 'None':
                if original_output == None:
                    original_output = self.backbone_model(*args, **kwargs)
                output = original_output
            elif parallel_module[0] in self.addable_adapter_like_delta_type:
                if original_output == None:
                    original_output = self.backbone_model(*args, **kwargs)
                output = self.parallel_modules[idx](original_output)

            else:
                # print(type(self.parallel_modules[idx]))
                output = self.parallel_modules[idx](*args, **kwargs)
            add_hiddens(output)

        assert self._alphas.shape[0] == len(hidden_states_list)
        new_hidden_states = torch.sum(
            F.softmax(self._alphas, dim=-1) * torch.stack(hidden_states_list, dim=-1), dim=-1)

        if isinstance(original_output, tuple):
            output = (new_hidden_states,) + original_output[1:]
        elif isinstance(original_output, torch.Tensor):
            output = new_hidden_states
        else:
            raise TypeError

        return output


    def find_key(self, key: Union[str, re.Pattern], target_list: List[str], only_tail=True):
        if key == '' and target_list == ['']:
            return True
        if self.common_structure and not hasattr(self, 'structure_mapping'):

            self.structure_mapping = CommonStructureMap.load(
                self.backbone_model)
        if self.common_structure:
            key = self.structure_mapping.transform(key, strict=False)
        if not key:
            return False
        try:
            if isinstance(key, re.Pattern):  # TODO: unit test needed ERROR
                if only_tail:
                    return endswith_in_regex(key, target_list)
                else:
                    return substring_in_regex(key, target_list)
            else:
                if only_tail:
                    return endswith_in(key, target_list)
                else:
                    return substring_in(key, target_list)
        except:
            from IPython import embed
            embed(header="exception")

    def find_module(self, root_module: nn.Module, key: str):
        if key == '':
            return root_module, 'new_module_for_replacing', root_module
        sub_keys = key.split(".")
        parent_module = root_module
        for sub_key in sub_keys[:-1]:
            parent_module = getattr(parent_module, sub_key)
        module = getattr(parent_module, sub_keys[-1])
        return parent_module, sub_keys[-1], module


class S3Delta(DeltaBase):
    default_modified_modules = [""]
    unfrozen_modules = [
        "deltas",
        "layer_norm",
        "final_layer_norm"
    ]

    def __init__(self,
                 backbone_model: nn.Module,
                 modified_configs: dict,
                 unfrozen_modules: Optional[List[str]] = None,
                 common_structure=False,
                 ):
        modified_modules = list(modified_configs.keys())

        DeltaBase.__init__(self,
                           backbone_model,
                           modified_modules=modified_modules,
                           unfrozen_modules=unfrozen_modules,
                           common_structure=common_structure,)
        self.new_module_list = []
        backbone = self.backbone_model
        self.modified_configs = modified_configs
        self.plm_total_params = sum(p.numel() for p in backbone.parameters())
        for key, _ in backbone.named_modules():
            if self.find_key(key, modified_modules):
                print("find key", key)
                self.update_module(backbone, key)

        self._pseudo_data_to_instantiate(backbone)
        for new_module, delta_names in self.new_module_list:
            setattr(new_module._alphas, 'delta_name_info', delta_names)
            for delta_module in new_module.delta_modules:
                if delta_module[1] is not None:
                    if isinstance(delta_module[1],DeltaBase):
                        delta_module[1].mark_as_delta()
                    elif isinstance(delta_module[1],nn.Module):
                        for p in delta_module[1].parameters():
                            setattr(p, "_is_delta", True)
                    else:
                        raise TypeError
        self.new_module_list = None

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
            replacedmodule, common_structure=self.common_structure)
        self.new_module_list.append((new_module, list(modified_config.keys())))
        for parallel_module_name in modified_config:
            # print(parallel_module_name)
            new_module.add_delta_from_config(parallel_module_name,
                                             (globals().get(f'{parallel_module_name}Config'))(**(modified_config[parallel_module_name])),)
        setattr(parent_module, children_name, new_module)
    # def mark_as_delta(self, module: nn.Module = None,):
    #     if module is None:
    #         module = self
    #     for n, p in module.named_parameters():
    #         if n.find('delta') != -1:
    #             setattr(p, "_is_delta", True)

    def num_backbone_parameters(self):
        return self.plm_total_params

    def freeze_not_delta_parameter(self, module: nn.Module):

        for n, p in module.named_parameters():
            if hasattr(p, "_is_delta") and getattr(p, "_is_delta") == True:
                p.requires_grad = True
            else:
                p.requires_grad = False
        return

    def named_trainable_parameters_without_alphas(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n.find('alphas') == -1 and p.requires_grad == True:
                yield n, p
            else:
                continue

    def trainable_parameters_without_alphas(self, model: nn.Module):
        for n, p in self.named_trainable_parameters_without_alphas(model):
            yield p

    def named_trainable_alphas(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n.find('alphas') != -1 and p.requires_grad == True:
                yield n, p
            else:
                continue
