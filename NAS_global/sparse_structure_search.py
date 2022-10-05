from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def hard_forward_structure(module_params, max_num_param):
    total_param = 0
    ret = [False for _ in range(len(module_params))]
    sorted_module_params_ids = sorted(list(range(len(module_params))),key = lambda x:module_params[x]['p_delta'],reverse=True)
    for sorted_module_params_id in sorted_module_params_ids:
        if total_param + module_params[sorted_module_params_id]['num_param'] <= max_num_param:
            total_param += module_params[sorted_module_params_id]['num_param']
            ret[sorted_module_params_id] = True
        else:
            threshold_p = module_params[sorted_module_params_id]['p_delta']
    return ret
    
@torch.no_grad()
def get_sparse_structure_p(args, model: nn.Module, delta_model):
    delta_model.do_sampling(us=None,random=False,hard_forward=args.hard_forward,training=False)
    options = hard_forward_structure(delta_model.new_module_params,args.max_num_param)
    new_structure_configs = [{} for _ in range(delta_model.modified_configs_length)]

    for idx,option in enumerate(options):
        if option:
            module_param = delta_model.new_module_params[idx]
            
            new_structure_config = new_structure_configs[module_param['modified_configs_length']]
            param_name = new_structure_config.get(module_param['key'], {})
            new_structure_config[module_param['key']] = param_name
            insert_method = param_name.get(module_param['insert_method'], {})
            param_name[module_param['insert_method']] = insert_method
            insert_method[module_param['delta_type']
                          ] = module_param['origin_params_dict']
    return new_structure_configs
    
def random_structure(module_params, max_num_param):
    def choice(remaining_param,module_params,chosen_list):
        choice_list = []
        for idx, module_param in enumerate(module_params):
            if module_param['num_param']<=remaining_param:
                choice_list.append(idx)
        if len(choice_list)==0:
            return None
        else:
            while True:
                ret_c = np.random.choice(choice_list)
                if chosen_list[ret_c] == False:
                    break
            return ret_c

    total_param = 0
    ret = [False for _ in range(len(module_params))]

    while True:
        c = choice(max_num_param-total_param,module_params,ret)
        if c is None:
            break
        else:
            ret[c] = True
            total_param+=module_params[c]['num_param']
    logger.info(f'total_param={total_param}')
    return ret

@torch.no_grad()
def get_sparse_structure_random(args, model: nn.Module, delta_model):
    # delta_model.do_sampling(us=None,random=False,hard_forward=args.hard_forward,training=False)
    options = random_structure(delta_model.new_module_params,args.max_num_param)
    new_structure_configs = [{} for _ in range(delta_model.modified_configs_length)]

    for idx,option in enumerate(options):
        if option:
            module_param = delta_model.new_module_params[idx]
            
            new_structure_config = new_structure_configs[module_param['modified_configs_length']]
            param_name = new_structure_config.get(module_param['key'], {})
            new_structure_config[module_param['key']] = param_name
            insert_method = param_name.get(module_param['insert_method'], {})
            param_name[module_param['insert_method']] = insert_method
            insert_method[module_param['delta_type']
                          ] = module_param['origin_params_dict']
    return new_structure_configs
    
def get_sparse_structure(args, model: nn.Module, delta_model):
    # Sort by p
    if args.detemining_structure_strategy == 'p':
        logger.info(f'detemining_structure_strategy: p')
        return get_sparse_structure_p(args, model, delta_model)
    # random sampling
    elif args.detemining_structure_strategy == 'random':
        logger.info(f'detemining_structure_strategy: random')
        return get_sparse_structure_random(args, model, delta_model)