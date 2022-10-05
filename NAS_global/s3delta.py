from audioop import bias
from typing import List, Optional, Union
import numpy as np
from scipy.optimize import fsolve
from collections import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from opendelta import BaseDeltaConfig
from opendelta.basemodel import DeltaBase
from opendelta.delta_models.adapter import (AdapterConfig, AdapterLayer,
                                            AdapterModel)
from opendelta.delta_models.bitfit import BiasLayer, BitFitConfig, BitFitModel
from opendelta.delta_models.compacter import (CompacterConfig, CompacterModel,
                                              HyperComplexAdapterLayer)
from opendelta.delta_models.layers.activations import Activations
from opendelta.delta_models.lora import LoraConfig, LoraModel
from opendelta.delta_models.low_rank_adapter import (LowRankAdapter,
                                                     LowRankAdapterConfig,
                                                     LowRankAdapterModel)
from opendelta.delta_models.prefix import (PrefixConfig, PrefixLayerT5,
                                           PrefixModel)
from opendelta.delta_models.soft_prompt import SoftPromptConfig,SoftPromptLayer
from opendelta.utils.cuda import get_device
from opendelta.utils.signature import get_arg_names, get_arg_names_inside_func

from opendelta.utils.utils import *
from opendelta.utils.visualization import Visualization
from transformers.models.t5.modeling_t5 import T5Attention

from delta_layers import (AdapterSequentialLayer, BitFitParallelLayer, BitFitSequentialLayer,
                          HyperComplexAdapterSequentialLayer,
                          LoraParallelLayer, LowRankAdapterSequentialLayer, T5LayerNormParalleyLayer,
                          NoneConfig, LNFitConfig, BitFitParallelConfig, BitFitSequentialConfig)
from sparse_structure_search import hard_forward_structure
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EPS = 1e-7


def print_alphas(delta_model):
    ret = '='*25+'alphas'+'='*25+'\n'
    for param in delta_model.new_module_params:
        ret += f"|{param['key']:<45s}| {param['insert_method']:<12s}|{param['delta_type']:<17s}|{param['num_param']:<.7f}  |{param['alpha'].item():<.5f}  |{param['p_delta']:<.5f}  \n"
    ret += '='*56+'\n'
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


class MixLayerParallel(nn.Module):

    addable_delta_type = ['None', 'Adapter', 'BitFitParallel', 'BitFitSequential',
                          'Compacter', 'Lora', 'LowRankAdapter', 'LNFit','SoftPrompt']
    addable_adapter_like_delta_type = [
        'Adapter', 'Compacter', 'LowRankAdapter']
    type2class = {
        'Adapter': AdapterSequentialLayer,
        'Compacter': HyperComplexAdapterSequentialLayer,
        'LowRankAdapter': LowRankAdapterSequentialLayer,
        'BitFitParallel': BitFitParallelLayer,
        'BitFitSequential': BitFitSequentialLayer,
        'Lora': LoraParallelLayer,
        'LNFit': T5LayerNormParalleyLayer,
        'SoftPrompt':SoftPromptLayer
    }

    def __init__(self,
                 backbone_model: nn.Module,
                 _parallel_alphas=None,
                 _sequential_alphas=None,
                 args=None,
                 common_structure: Optional[bool] = None
                 ):
        nn.Module.__init__(self)

        self.backbone_model = backbone_model
        self._parallel_alphas = _parallel_alphas
        self._sequential_alphas = _sequential_alphas
        self.args = args
        self.common_structure = common_structure

        self.parallel_modules = nn.ModuleList()
        self.parallel_alphas = nn.ParameterList()
        self.parallel_modules_params = []
        self.num_parallel_modules = 0

        self.sequential_modules = nn.ModuleList()
        self.sequential_alphas = nn.ParameterList()
        self.sequential_modules_params = []
        self.num_sequential_modules = 0

        self.gb_sample = None

        self.is_prompt = False

    def add_delta_from_config(self,
                              config: Union[BaseDeltaConfig, dict],
                              modified_configs_length: int,
                              key: str,
                              insert_method: str,
                              delta_type: str,
                              origin_params_dict: dict):

        assert insert_method in ["parallel", "sequential"]
        assert delta_type in self.addable_delta_type, f'delta_type should be in {self.addable_delta_type}, but got {delta_type}'

        if delta_type == 'Lora':  # Lora
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
        elif delta_type == 'SoftPrompt':  # softprompt
            self.is_prompt = True
            layertype = self.type2class[delta_type]
            arg_names = get_arg_names(layertype)
            args = {}
            if 'device' in arg_names:
                args['device'] = get_device(self.backbone_model)
            for arg_name in arg_names:
                if arg_name in config.__dict__:
                    args[arg_name] = getattr(config, arg_name)
            args['raw_embedding'] = self.backbone_model.get_input_embeddings()
            new_module = layertype(**args)

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
        else:  # None
            new_module = None

        modules_list = getattr(self, f"{insert_method}_modules")
        modules_list.append(new_module)

        alpha = nn.Parameter(torch.zeros(1))
        setattr(alpha, 'delta_type', delta_type)
        alpha_list = getattr(self, f"{insert_method}_alphas")
        alpha_list.append(alpha)

        modules_params = getattr(self, f"{insert_method}_modules_params")
        params = {
            'module': new_module,
            'alpha': alpha,
            'z_delta': 1,
            'modified_configs_length': modified_configs_length,  # 0, 1, ...
            'key': key,  # module name
            'insert_method': insert_method,  # parallel, sequential
            'delta_type': delta_type,  # delta(lora,adapter)
            'origin_params_dict': origin_params_dict,  # config of specific delta method
        }
        modules_params.append(params)

        setattr(self, f"num_{insert_method}_modules", getattr(
            self, f"num_{insert_method}_modules")+1)

    def forward(self, *args, **kwargs):
        if self.is_prompt:
            args, kwargs = self.sequential_modules_params[0]['module'](*args, **kwargs)
            return self.backbone_model(*args, **kwargs)

        original_output = self.backbone_model(*args, **kwargs)

        def get_hiddens(output):
            if isinstance(output, tuple):
                hiddens = output[0]
            elif isinstance(output, torch.Tensor):
                hiddens = output
            else:
                raise TypeError
            return hiddens

        # parallell output
        delta_output = 0
        for idx, parallel_modules_param in enumerate(self.parallel_modules_params):
            if parallel_modules_param['insert_method'] == 'None':
                output = torch.zeros_like(get_hiddens(original_output))
            else:
                output = parallel_modules_param['module'](*args, **kwargs)
            output_dim = get_hiddens(output).shape[-1]
            setattr(parallel_modules_param['alpha'], 'output_dim', output_dim)

            if not self.args.for_baseline:
                rate = parallel_modules_param['z_delta']
            else:
                rate = 1
            if rate > 0:
                delta_output = rate * output + delta_output

        if isinstance(original_output, tuple):
            parallel_output = (delta_output +
                               original_output[0],) + original_output[1:]
        elif isinstance(original_output, torch.Tensor):
            parallel_output = delta_output + original_output
        elif isinstance(original_output,dict) :
            print(list(original_output.keys())[0])
            original_output[list(original_output.keys())[0]] += delta_output 
            parallel_output = original_output
        else:
            print(type(original_output[0]))
            print(isinstance(original_output,dict))
            raise TypeError

        # sequential output
        delta_output = 0
        for idx, sequential_modules_param in enumerate(self.sequential_modules_params):
            if sequential_modules_param['insert_method'] == 'None':
                output = torch.zeros_like(get_hiddens(parallel_output))
            else:
                # print(sequential_modules_param['module'])
                output = sequential_modules_param['module'](parallel_output)

            output_dim = get_hiddens(output).shape[-1]
            setattr(
                sequential_modules_param['alpha'], 'output_dim', output_dim)

            if not self.args.for_baseline:
                rate = sequential_modules_param['z_delta']
            else:
                rate = 1
            if rate > 0:
                delta_output = rate * get_hiddens(output) + delta_output

        if isinstance(parallel_output, tuple):
            sequential_output = (
                delta_output+parallel_output[0],) + parallel_output[1:]
        elif isinstance(parallel_output, torch.Tensor):
            sequential_output = delta_output+parallel_output
        elif isinstance(original_output,dict) :
            print(list(original_output.keys())[0])
            original_output[list(original_output.keys())[0]] += delta_output 
            parallel_output = original_output
        else:
            raise TypeError

        return sequential_output


class S3Delta(DeltaBase):
    default_modified_modules = [""]
    unfrozen_modules = [
        "deltas"
    ]

    def __init__(self,
                 backbone_model: nn.Module,
                 args,
                 unfrozen_modules: Optional[List[str]] = None,
                 common_structure=False,
                 ):

        DeltaBase.__init__(self,
                           backbone_model,
                           modified_modules=[],
                           unfrozen_modules=unfrozen_modules,
                           common_structure=common_structure,)
        self.modified_configs_length = 0
        self.args = args
        self.new_module_list = []
        self.new_module_params = []
        self.modified_configs_list = []
        self.backbone_model_params = sum(p.numel()
                                         for p in self.backbone_model.parameters())/1024**2
        self.last_tau = 1
        self.last_bia = 0

    def modify_backbone_from_config(self, modified_configs: dict):
        self.modified_configs = modified_configs
        self.modified_configs_list.append(modified_configs)

        modified_modules = list(modified_configs.keys())
        for key, _ in self.backbone_model.named_modules():
            if self.find_key(key, modified_modules):
                # print("find key", key)
                modified_config = self.find_config(key)
                self.update_module(self.backbone_model, key, modified_config)

        self._pseudo_data_to_instantiate(self.backbone_model)
        self.mark_as_delta()
        self.calculate_params_num()

        self.new_module_list = []

        self.modified_configs_length += 1

    def find_config(self, key):
        for i in self.modified_configs:
            if self.find_key(key, [i]):
                return self.modified_configs[i]

    def update_module(self, module: nn.Module, key: str, modified_config):
        parent_ref, children_name, child_ref = self.find_module(module, key)
        self.replace_module(parent_ref, children_name,
                            child_ref, key, modified_config)

    def replace_module(self, parent_module: nn.Module, children_name: str, children_module: Optional[nn.Module] = None, key=None, modified_config=None):
        new_module = MixLayerParallel(
            children_module, args=self.args, common_structure=self.common_structure)
        self.new_module_list.append(new_module)
        for insert_method in modified_config:
            for delta_module_name in modified_config[insert_method]:
                new_module.add_delta_from_config((globals().get(f'{delta_module_name}Config'))(
                    **(modified_config[insert_method][delta_module_name])), self.modified_configs_length, key, insert_method, delta_module_name, modified_config[insert_method][delta_module_name])
        self.new_module_params.extend(new_module.parallel_modules_params)
        self.new_module_params.extend(new_module.sequential_modules_params)
        setattr(parent_module, children_name, new_module)

    def mark_as_delta(self, module: nn.Module = None,):

        if module == None:
            module = self
        for new_module in module.new_module_list:
            for parallel_module in new_module.parallel_modules:
                if parallel_module is not None:
                    for p in parallel_module.parameters():
                        setattr(p, "_is_delta", True)
            for sequential_module in new_module.sequential_modules:
                if sequential_module is not None:
                    for p in sequential_module.parameters():
                        setattr(p, "_is_delta", True)
            for alpha in new_module.parallel_alphas:
                setattr(alpha, "_is_delta", True)
            for alpha in new_module.sequential_alphas:
                setattr(alpha, "_is_delta", True)

    def calculate_params_num(self):
        for idx, param in enumerate(self.new_module_params):
            param['num_param'] = sum(
                p.numel() for p in param['module'].parameters() if p.requires_grad)/1024**2

    def calculate_tau_and_theta(self, new_module_params, training=True):
        alphas = np.array([p['alpha'].item() for p in new_module_params])
        param_nums = np.array([p['num_param']
                              for p in new_module_params])

        e_alphas = np.exp((alphas - np.max(alphas)))
        softmax_alphas = e_alphas / e_alphas.sum()
        if self.args.sparsity_decay_step > 0 and training:
            max_num_param = self.args.sparsity_decay_max_num_param
        else:
            max_num_param = self.args.max_num_param
        theta = max_num_param/(softmax_alphas*param_nums).sum()
        if theta*np.max(softmax_alphas) <= 1.0:  # tau* <= 1
            return 1, 1, theta
        else:
            def func(tau):
                tau = np.abs(tau)
                e_x = np.exp((alphas - np.max(alphas))/tau)
                softmax_x = e_x / e_x.sum()
                ret = (softmax_x*param_nums).sum() / \
                    max_num_param - 1/e_x.sum() - EPS
                return ret
            org_tau = np.abs(fsolve(func, self.last_tau)[0])
            tau = min(max(org_tau, 1), 10000)  # prevent overflow

            e_alphas = np.exp((alphas - np.max(alphas))/tau)
            softmax_alphas = e_alphas / e_alphas.sum()
            theta = max_num_param/(softmax_alphas*param_nums).sum()

            return org_tau, tau, theta

    def calculate_bia_for_shifting(self, new_module_params, training=True):
        alphas = np.array([p['alpha'].item() for p in new_module_params])
        param_nums = np.array([p['num_param'] for p in new_module_params])

        if self.args.sparsity_decay_step > 0 and training:
            max_num_param = self.args.sparsity_decay_max_num_param
        else:
            max_num_param = self.args.max_num_param

        def func(bia):
            x = (alphas - bia)/self.args.beta
            x[x < -200] = -200
            sigmoid_shifted_alphas = 1/(1+np.exp(-x))
            ret = (sigmoid_shifted_alphas*param_nums).sum() - \
                max_num_param*(1 - EPS)
            return ret
        bia_res = fsolve(func, self.last_bia)[0]

        return bia_res

    def do_sampling(self, us, random, hard_forward, training, using_last_us=False):

        if us is not None and random == False:
            raise NotImplementedError
        if us is not None and using_last_us == True:
            raise NotImplementedError

        # calculate tau* and theta
        if self.args.global_strategy == "tau_theta":
            if using_last_us:
                us = self.last_us
                tau = self.last_tau
                theta = self.last_theta
            else:
                org_tau, tau, theta = self.calculate_tau_and_theta(
                    self.new_module_params, training=training)

            # calculate p_delta
            alphas = torch.cat([p['alpha']
                               for p in self.new_module_params], dim=0)
            p_deltas = theta*F.softmax(alphas/tau, dim=-1)
            p_deltas[p_deltas >= 1] = 1-EPS

        # calculate bias for shifting
        elif self.args.global_strategy == "shift":
            if using_last_us:
                us = self.last_us
                bia = self.last_bia
            else:
                bia = self.calculate_bia_for_shifting(
                    self.new_module_params, training=training)

            alphas = torch.cat([p['alpha']
                               for p in self.new_module_params], dim=0)

            x = (alphas - bia)/self.args.beta
            x[x < -200] = -200

            p_deltas = torch.sigmoid(x)
            dp = torch.zeros_like(p_deltas)
            dp[p_deltas == 1] = -EPS
            dp[p_deltas == 0] = EPS
            p_deltas = p_deltas+dp
            if self.args.shift_detach:
                p_deltas = p_deltas/p_deltas.sum()*p_deltas.sum().detach()

        elif self.args.global_strategy == "l0-norm":
            if using_last_us:
                us = self.last_us
            alphas = torch.cat([p['alpha']
                               for p in self.new_module_params], dim=0)
            p_deltas = torch.sigmoid((alphas)/self.args.beta)

        for idx, param in enumerate(self.new_module_params):
            param['p_delta'] = p_deltas[idx]

        num = len(self.new_module_params)

        if us is not None:
            us[us == 0] = EPS
            us[us == 1] = 1-EPS
        if training:
            if self.args.using_gumbel:
                if us is None:
                    if random:
                        us = torch.rand(num).to(p_deltas.device)
                        s_deltas = torch.sigmoid(
                            torch.log(us*p_deltas/(1-us)/(1-p_deltas)))
                    else:
                        s_deltas = p_deltas
                else:
                    us = us.to(p_deltas.device)
                    s_deltas = torch.sigmoid(
                        torch.log(us*p_deltas/(1-us)/(1-p_deltas)))
            else:
                s_deltas = p_deltas

        if hard_forward:
            if training:
                s_hat_deltas = s_deltas*(self.args.r-self.args.l)+self.args.l
                z_deltas = torch.min(torch.ones(num).to(s_hat_deltas.device), torch.max(
                    torch.zeros(num).to(s_hat_deltas.device), s_hat_deltas))
                # s_hat_delta = torch.ones_like(p_deltas)
                z_deltas = torch.ones_like(
                    z_deltas) - z_deltas.detach() + z_deltas
                index = s_hat_deltas < (self.args.r+self.args.l)/2
                z_deltas[index] = 0.0
                if torch.sum(z_deltas) == 0.0:
                    logger.info('none has been chosen, use largest p_delta.')
                    m = torch.argmax(p_deltas)
                    z_deltas[m] = 1 - p_deltas[m].detach() + p_deltas[m]
            else:
                options = hard_forward_structure(
                    self.new_module_params, self.args.max_num_param)
                z_deltas = [(1.0 if option else 0.0) for option in options]

        else:  # soft
            if training:
                s_hat_deltas = s_deltas*(self.args.r-self.args.l)+self.args.l
                z_deltas = torch.min(torch.ones(num).to(s_hat_deltas.device), torch.max(
                    torch.zeros(num).to(s_hat_deltas.device), s_hat_deltas))
                if torch.sum(z_deltas) == 0:
                    m = torch.argmax(p_deltas)
                    z_deltas[m] = p_deltas[m]
            else:
                s_hat_deltas = p_deltas*(self.args.r-self.args.l)+self.args.l
                z_deltas = torch.min(torch.ones(num).to(s_hat_deltas.device), torch.max(
                    torch.zeros(num).to(s_hat_deltas.device), s_hat_deltas))

        for idx, param in enumerate(self.new_module_params):
            param['z_delta'] = z_deltas[idx]

        if self.args.global_strategy == "tau_theta":
            self.last_tau = tau
            self.last_theta = theta
            self.last_us = us
        elif self.args.global_strategy == "shift":
            self.last_bia = bia
            self.last_us = us
        elif self.args.global_strategy == "l0-norm":
            self.last_us = us

    def num_backbone_parameters(self):
        return self.backbone_model_params
