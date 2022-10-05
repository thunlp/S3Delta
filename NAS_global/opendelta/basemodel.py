

import os
from opendelta.delta_configs import BaseDeltaConfig
from opendelta.utils.model_md5 import gen_model_hash
from opendelta.utils.signature import get_arg_names, signature
from typing import Optional, Union
from opendelta.utils.cuda import get_device
from opendelta.utils.utils import *
import torch.nn as nn
import torch
from functools import wraps
from decorator import decorate
from opendelta.utils.structure_mapping import transform
from transformers.file_utils import PushToHubMixin
from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled
from opendelta import SaveLoadMixin
from opendelta import logging
from opendelta.utils.structure_mapping import CommonStructureMap

logger = logging.get_logger(__name__)

def is_leaf_module(module):
    r"""Whether the module is a leaf module
    """
    try:
        return len([n for n,_ in module.named_children()]) == 0
    except:
        from IPython import embed
        embed()

def non_module_param(module: nn.Module):
    module_names = [n for n, _ in module.named_modules()]
    ret = []
    for n, p in module.named_parameters():
        if not is_child_key(n, module_names):
            ret.append((n,p))
    return ret




class DeltaBase(nn.Module, SaveLoadMixin):
    r"""This is the base class for all delta models. It provides four simple but effective functionalities 
    for building the delta model:

        #. addressing a module inside the backbone model using a minimal description key. 
        #. provide the interface for modifying and inserting model which keeps the docs/IO the same as the module 
           before modification.
        #. pass a pseudo input to determine the inter dimension of the delta models. 
        #. freeze a part of model parameters according to key. 
        
        It also provides unified interface for model loading and saving. 

    Class attributes (overridden by derived classes):
    
    - delta_type (:obj:`str`) the name of the delta modules, used to create the correct [`~opendelta.AutoDeltaModel`].
    - config_class (:obj:`BaseDeltaConfig`) The corresponding config model 


    Args:
        backbone_model (:obj:`nn.Module`, *required*)  backbone model that the delta models are build opon. The modification to the 
            backbone model are in place.
        modified_modules (:obj:`List[str]`, *optional*, default to :obj:`None`) The modules are subjected to update. 
            
            .. note::
                leave this argument :obj:`None` will make the delta model return to the default setting, which add the delta
                models to the position experimented the paper. In this setting, the common structure mapping is loaded to 
                addressing the corresponding modules.

        registraction_name (:obj:`str`, *optional*, default to :string:`"deltas"`) The root name of the delta models when
            attached to the backbone model. 
        common_structure (:obj:`bool`, *optional* ,default to :obj:`None`) Whether use the common structure mapping to 


    """
    delta_type = ""
    config_class = BaseDeltaConfig
    default_unfrozen_modules = ["deltas"]
    def __init__(self, 
                 backbone_model: nn.Module,
                 modified_modules: Optional[List[str]] = None,
                 unfrozen_modules: Optional[List[str]] = None,
                 common_structure=False,
                 ):
        nn.Module.__init__(self)
        # register the backbone model after init using self.__dict__ method to avoid adding backbone_model
        # to the modules of the delta model.
        self.__dict__["backbone_model"] = backbone_model
        if modified_modules  is None:
            self.modified_modules = self.default_modified_modules
            self.common_structure = True
        else:
            self.modified_modules = modified_modules
            self.common_structure = common_structure
        if self.common_structure:
            self.structure_mapping = CommonStructureMap.load(self.backbone_model)
        else:
            self.structure_mapping = None
        if unfrozen_modules is None:
            self.unfrozen_modules = self.default_unfrozen_modules
        if self.common_structure and self.structure_mapping is None:
            raise RuntimeError("Using common structure but the structure mapping is None")
        
    def forward(self, *args, **kwargs) -> "RuntimeError":
        r""" 
            .. warning::

                Removed method. As the model is a delta model, which should be attached to a backbone model \
                and can't forward any data by itself. Please using the backbone model's forward function \
                after attach the delta model to the backbone.
        """
        raise RuntimeError("This is a delta model, which should be attached to a backbone model \
            and can't forward any data by itself. Please using the backbone model's forward function \
            after attach the delta model to the backbone. ")

    @classmethod
    def from_config(cls, config: Union[BaseDeltaConfig, dict], backbone_model: nn.Module, check_hash=True, **kwargs):
        r"""Initialize a delta model from a config object or a dict containing the configs. To temperarily change
        a value in the config, pass it through kwargs. If the config has a backbone model's hash, which means it is
        a finetuned delta model's config, then we will compare the hash in the config and the newly caculated to ensure
        the finedtuned delta model is trained on the passed backbone_model. Pass `check_hash=False` to disable the
        checking.

        Args:
            config (:obj:`BaseDeltaConfig` or `dict`) A config object or a dict that contains the necessary value to 
                            initialize the delta model.
            backbone_model (:obj:`nn.Module`) A pytorch module that will be pass into the delta model as the backbone 
                    model. modifications will be made in place in the backbone model.
            check_hash (:obj:`bool`, default to `True`) Whether to check hash of the backbone model and the config's 
                            backbone hash. 
            kwargs: Any configurations that are passed to update the config object. #TODO unit test needed.
        """
        supported_keys = get_arg_names(cls.__init__) + get_arg_names(DeltaBase.__init__)
        config_dict = config.to_dict()
        for key in list(config_dict.keys()):
            if key not in supported_keys:
                config_dict.pop(key)
        return cls(backbone_model, **config_dict)


    def add_all_delta_to_backbone(self, 
                 backbone: nn.Module, 
                 modified_modules: List[str],
                ) -> nn.Module:
        r"""The main function to add delta models to the backbone model based on the :obj:`modified_modules`.
        

        Args:
            backbone_model (:obj:`nn.Module`, *required*)  backbone model that the delta models are build opon. The 
                modification to the backbone model are in place.
            modified_modules (:obj:`List[str]`, *optional*, default to :obj:`None`) The modules are subjected to update. 
                leave this argument :obj:`None` will make the delta model return to the default setting, which add the delta
                models to the position experimented the paper. In this setting, the common structure mapping is loaded to 
                addressing the corresponding modules.

        Returns:
            :obj:`nn.Module` The modified backbone model.

        """
        self.plm_total_params = sum(p.numel() for p in backbone.parameters())
        # create a new key list to avoid recursion.
        backbone_key_list = [key for key, _ in backbone.named_modules()] 
        # print(backbone_key_list)
        for key in backbone_key_list:
            if self.find_key(key, modified_modules): #TODO have bugs when commonstructure has a virtual node and it's refered
                # print("find key",key)
                self.update_module(backbone, key)
        if (not hasattr(backbone,'pseudo')) or backbone.pseudo == True:
            self._pseudo_data_to_instantiate(backbone)
        # mark the paratmers that are the delta parameters for easily 
        # extracting the delta_paramters.
        # This is important if the delta parameters are contained in the
        # original models parameters
        self.mark_as_delta()
        return backbone
    

    
    def mark_as_delta(self, module: nn.Module=None,):
        r""" Mark :obj:`module`'s all parameters as delta parameters by setting a :string:`_is_delta`  attribute to each of them.
        Generally, it is used after creating the delta modules. By leaving module to :obj:`None`, it will mark all the parameters in the 
        delta model as :string:`_is_delta`.

        Args:
            module (:obj:`nn.Module`): The module to mark as delta.
        """
        if module is None:
            module=self # all the parameters in the delta model.
        for p in module.parameters():
            setattr(p, "_is_delta", True)
    
    def update_module(self, module: nn.Module, key: str):
        r"""Update a module specified by :obj:`key`. The method is reimplemented in each specific delta model. 
        """
        raise NotImplementedError
    
    
    def freeze_module(self,
                      module: Optional[nn.Module] = None, 
                      exclude: Optional[List[str]] = None, 
                      set_state_dict: Optional[bool]=True, 
                      ):
        r"""Freeze the parameters of plm. Leave the parameters in exclude untouched.
        deltas module is filtered with `_is_delta` attributes because it may have parameter sharing to the main 
        model, (e.g., bias term)

        Args:
            module (:obj:`nn.Module`, *optional*, default to :obj:`None`) The module of which some parts are frozen.
                If left with :obj:`None`, the function will the self.backbone_model as the module to be frozen. 
            exclude (:obj:`List[str]`, *optional*, default to :string:`["deltas"]`) The parameters that don't need to 
                be freezed. Default to all the delta parameters.
            set_state_dict (:obj:`bool`, *optional*, default to :obj:`True`) Whether setting the backbone model's state
                dict to all the parameters that still need grad.
            prefix (:obj:`str`, *optional*, default to :string:`""`) A parameters that are used for recursive frozen. 
                Should not be changed by passing argument other than :string:`""`.
        
        """
        if exclude is None:
            exclude = self.unfrozen_modules

        if module is None:
            module = self.backbone_model
        self._freeze_module_recursive(module, exclude, "")           # modify the active state dict that still need grad
        if set_state_dict:
            self.set_active_state_dict(module)

    def _freeze_module_recursive(self,
                      module: Optional[nn.Module] = None, 
                      exclude: Optional[List[str]] = None,
                      prefix=""):
        r"""Freeze the parameters of plm. Leave the parameters in exclude untouched.
        deltas module is filtered with `_is_delta` attributes because it may have parameter sharing to the main 
        model, (e.g., bias term)

        Args:
            module (:obj:`nn.Module`, *optional*, default to :obj:`None`) The module of which some parts are frozen.
                If left with :obj:`None`, the function will the self.backbone_model as the module to be frozen. 
            exclude (:obj:`List[str]`, *optional*, default to :string:`["deltas"]`) The parameters that don't need to 
                be freezed. Default to all the delta parameters.
            set_state_dict (:obj:`bool`, *optional*, default to :obj:`True`) Whether setting the backbone model's state
                dict to all the parameters that still need grad.
            prefix (:obj:`str`, *optional*, default to :string:`""`) A parameters that are used for recursive frozen. 
                Should not be changed by passing argument other than :string:`""`.
        
        """

        if is_leaf_module(module):
            for n, p in module.named_parameters():
                if self.find_key(".".join([prefix,n]), exclude, only_tail=True):
                    continue
                if "deltas" not in exclude or (not (hasattr(p, "_is_delta") and getattr(p, "_is_delta"))):
                    p.requires_grad = False
            return 
        else:
            for n, c in module.named_children():
                if self.find_key(".".join([prefix,n]), exclude, only_tail=True): # if found, untouch the parameters
                    continue
                else: # firstly freeze the non module params, then go deeper.
                    params = non_module_param(module)
                    for n, p in params:
                        if "deltas" not in exclude or (not (hasattr(p, "_is_delta") and getattr(p, "_is_delta"))):
                            p.requires_grad = False
                    self._freeze_module_recursive(c, exclude=exclude, prefix=".".join([prefix,n]) )
        




    def find_key(self, key: Union[str, re.Pattern], target_list: List[str], only_tail=True):
        r"""Check whether any target string is in the key or in the tail of the key, i.e., 

        Returns: 
            :obj:`bool` True if the key matchs the target list.
        """
        if key=='' and target_list==['']:
            return True
        if self.common_structure:
            key = self.structure_mapping.transform(key, strict=False)
        if not key:
            return False
        try:
            if isinstance(key, re.Pattern): # TODO: unit test needed ERROR
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
            embed(header = "exception")

    def _pseudo_data_to_instantiate(self, module):
        r"""[NODOC] Create a pseudo_data into the module to know the dimemsion of each tensor in the computation graph.
        #TODO: To test more data input format, i.e. may need to pass more than inputs/decoder_input_ids.
        """
        device = get_device(module)
        pseudo_input = torch.tensor([[0,0]]).to(device)
        if "decoder_input_ids" in  signature(module.forward).args:
            module(pseudo_input, decoder_input_ids = pseudo_input)
        else:
            module(pseudo_input)

    def trainable_parameters_names(self, module: Optional[nn.Module]=None):
        r"""A small sugar function to return all the trainable parameter's name in the (by default, backbone) model.

        Args: 
            module (:obj:`nn.Module`): of which module we want to know the trainable paramemters' name.
        
        Returns:
            :obj:`List[str]`
        """
        if module is None:
            module = self.backbone_model
        return [n for n,p in module.named_parameters() if p.requires_grad]
    
    def frozen_parameters_names(self, module: Optional[nn.Module]=None):
        r"""A small sugar function to return all the frozen parameters' name in the (by default, backbone) model.

        Args: 
            module (:obj:`nn.Module`): of which module we want to know the frozen paramemters' name.
        
        Returns:
            :obj:`List[str]`
        """
        if module is None:
            module = self.backbone_model
        return [n for n,p in module.named_parameters() if not p.requires_grad]

    def trainable_parameters(self,module: Optional[nn.Module]=None):
        r"""A small sugar function to return all the frozen parameters in the (by default, backbone) model.

        Args: 
            module (:obj:`nn.Module`): of which module we want to know the frozen paramemters.
        
        Returns:
            :obj:`List[nn.Parameter]` 
        """
        if module is None:
            module = self
        return [p for n,p in module.named_parameters() if p.requires_grad]


    def num_trainable_parameters(self, module: Optional[nn.Module]=None):
        r"""A small sugar function to get the number of trainable parameter in the backbone model. Often used to 
        compute the trainable rate.

        Args: 
            module (:obj:`nn.Module`): of which module we want to know the number of trainable paramemters.
        
        Returns:
            :obj:`List[nn.Parameter]` 
        """
        if module is None:
            module = self
        pnum_tot = 0
        for param in module.parameters():
            if param.requires_grad:
                pnum_tot += param.numel()
        return pnum_tot
    
    def num_total_parameters(self, module: Optional[nn.Module]=None):
        r"""A small sugar function to get the number of trainable parameter in the backbone model. Often used to 
        compute the trainable rate.

        Args: 
            module (:obj:`nn.Module`): of which module we want to know the number of trainable paramemters.
        
        Returns:
            :obj:`List[nn.Parameter]` 
        """
        if module is None:
            module = self
        pnum_tot = 0
        for param in module.parameters():
            pnum_tot += param.numel()
        return pnum_tot
    
    # def num_frozen_parameters(self, module: Optional[nn.Module]=None):
    #     if module is None:
    #         module = self
    #     pnum_tot = 0
    #     for param in module.parameters():
    #         if not param.requires_grad:
    #             pnum_tot += param.numel()
    #     return pnum_tot

    def find_module(self, root_module: nn.Module, key:str):
        r"""Find the module using a key and the root module. Return both the parent reference, the child name and reference.

        Args:
            root_module (:obj:`root_module`) The root_module to find the sub module in
            key (:obj:`str`) The relative key to the root module. 

        Returns:
            (:obj:`nn.Module`, :obj:`str`, :obj:`nn.Module`): 
            * A reference to the parent module of the target module, mainly for substuting the target module. 
            * The key of the target module relevant to its parent module
            * Target module.
        """
        if key=='':
            return root_module,'new_module_for_replacing', root_module
        sub_keys = key.split(".")
        parent_module = root_module
        for sub_key in sub_keys[:-1]:
            parent_module = getattr(parent_module, sub_key)
        module = getattr(parent_module, sub_keys[-1])
        return parent_module, sub_keys[-1], module

    def replace_module(self,
                      parent_module: nn.Module, 
                      children_name: str,
                      replacedmodule: Optional[nn.Module]=None):
        r"""Replace a module using the reference of its parent module. This method will be reimplemented in different
        derived class if needed
        """
        raise NotImplementedError
    
    def modify_module(self, module: nn.Module):
        r"""Modify the inside parameteres of a module. This method will be reimplemented in different
        derived class if needed.
        """
        raise NotImplementedError

    def insert_sequential_module(self, module, pre_caller=None, post_caller=None, delta_module=None, name='delta'):
        r"""insert a module (previous not exists in the code base) before a module. Specifically, it modifies the forward 
        function of the original module to  firstly pass the arguments into the new module's forward function and then pass
        it into the original ones. The new module can also be inserted after the original module with similar mechanism. 

        When implementing the new module , researchers should be aware of the components of arguments of the original module's forward function.
        """
        def _caller(_org_func, _pre_caller, _post_caller,  *args, **kwargs):
            if _pre_caller is not None:
                args, kwargs = _pre_caller(*args, **kwargs)
            ret = _org_func(*args, **kwargs)
            if _post_caller is not None:
                ret = _post_caller(ret)
            return ret

        if hasattr(module.forward, "__wrapped__"):
            raise RuntimeWarning("The forward function might have been wrapped by a decorator, is it intended?")
        module.forward = decorate(module.forward, _caller, extras=(pre_caller, post_caller), kwsyntax=True) # decorator.decorate helps preserving the functions metadata (signature, etc.).
        if delta_module is not None:
            setattr(module, name, delta_module)

    

    def insert_parrellel_module(self, ):
        # TODO
        pass

    def set_active_state_dict(self, module: nn.Module):
        r"""modify the state_dict function of the model (by default, the backbone model) to return only the tunable part.

        Args:
            module (:obj:`nn.Module`): The module modified. The modification is in-place.
        """
        def _caller(_org_func, includes,  *args, **kwargs):
            state_dict = _org_func(*args, **kwargs)
            keys = list(state_dict.keys())
            for n  in keys:
                if n not in includes:
                    state_dict.pop(n)
            return state_dict
        includes = self.trainable_parameters_names(module) # use excludes will have trouble when the model have shared weights
        # print(includes, "grad:",self.backbone_model.plm.lm_head.weight.requires_grad)
        # exit()
        if hasattr(module.state_dict, "__wrapped__"):
            raise RuntimeWarning("The forward function might have been wrapped by a decorator, is it intended?")
        module.state_dict = decorate(module.state_dict, _caller, extras=(includes,), kwsyntax=True) # decorator.decorate helps preserving the functions metadata (signature, etc.).
    

    def create_config_from_model(self, ):
        r"""[NODOC] If the delta model was built by directly passing arguments, instead of passing a config object.
        create the config of the delta model for saving the delta model.
        """
        # common_attributes
        config = self.config_class()
        config_keys = signature(config.__init__)[0] + signature(super(self.config_class, config).__init__)[0]

        for key in config_keys:
            val = getattr(self, key) if hasattr(self, key) else None
            setattr(config, key, val)
        config.delta_type = self.delta_type
        self.config = config
        
    
    def _load_state_dict_into_backbone(self, backbone_model: nn.Module = None, state_dict: dict = {}):
        r"""[NODOC]
        """
        if backbone_model is None:
            backbone_model = self.backbone_model
        self.backbone_model.load_state_dict(state_dict, strict=False)
    
    def log(self, delta_ratio=False, trainable_ratio=True, visualization=True,file=None):
        r"""Log the result of applying delta. Possible Options are :string:`trainable_ratio`,
        :string:`visualization`, :string:`delta_ratio`.
        """

        if visualization:
            from opendelta import Visualization
            Visualization(self.backbone_model).structure_graph(file=file)
        if trainable_ratio:
            n_trainable = self.num_trainable_parameters(self.backbone_model)
            n_total = self.num_total_parameters(self.backbone_model)
            logger.info("Trainable Ratio: {:2f}%".format(n_trainable/n_total*100))
        if delta_ratio:
            n_delta = self.num_trainable_parameters()
            n_total = self.num_total_parameters(self.backbone_model)
            logger.info("Delta Parameter Ratio: {:2f}%".format(n_delta/n_total*100))

        






        
    


        

