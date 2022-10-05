import torch.autograd as autograd
import math
from torch.optim import Optimizer
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from posixpath import split
from examples_seq2seq import metrics
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from opendelta.utils.visualization import Visualization
from architect import Architect, get_num_trainable_parameters
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from s3delta import S3Delta, MixLayerParallel, named_weights, weights, alphas, named_alphas, print_alphas
from delta_layers import reset_param
import copy
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# class Sparsity_rate_schedule:
#     '''sparsity rate from all params to budget'''
#     def __init__(self,args):
#         self.args = args
#         self.total_params = args.total_params
#         self.org_max_num_param = args.max_num_param
#         self.args.sparsity_decay_max_num_param = args.total_params
#         self.max_step = args.sparsity_decay_step
#         self.steps = 0

#         self.org_rate = self.total_params/self.org_max_num_param - 1

#     def step(self):
#         # from org_rate+1 to 1
#         self.steps+=1
#         # linear
#         rate = self.org_rate*(1-self.steps/self.max_step) 
#         self.args.sparsity_decay_max_num_param = max((rate+1),1)*self.org_max_num_param
        
#     def get_sparsity_rate(self):
#         return self.args.sparsity_decay_max_num_param


class s3deltaTrainer:
    def __init__(self,
                 model: nn.Module, delta_model,
                 v_model, v_delta_model,
                 tokenizer,
                 compute_metrics,
                 data_info,
                 args,
                 train_dataloader,
                 eval_dataloader,
                 eval_dataloader_not_shuffle,
                 test_dataloader):
        
        self.model = model
        self.delta_model = delta_model

        self.v_model = v_model
        self.v_delta_model = v_delta_model

        self.tokenizer = tokenizer

        self.compute_metrics = compute_metrics
        self.data_info = data_info

        self.args = args

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.eval_dataloader_not_shuffle = eval_dataloader_not_shuffle
        self.test_dataloader = test_dataloader

        self.global_step = 0

        self.valid_step = args.valid_steps
        self.w_lr = args.learning_rate
        self.alpha_lr = args.alpha_learning_rate

        # max_step
        self.max_step = self.args.train_epochs * len(self.train_dataloader)

        # w_warmup_steps
        self.w_warmup_steps = self.args.w_warmup_steps

        # alpha_warmup_steps
        self.alpha_warmup_steps = self.args.alpha_warmup_steps

        self.set_optimizer()
        self.set_schedule()
        self.set_architect()

        # if self.args.sparsity_decay_step > 0 and not self.args.for_baseline:
        #     self.sparsity_rate_schedule = Sparsity_rate_schedule(self.args)

    def set_optimizer(self, if_print_w=False,if_print_alphas=False):
        # weights optimizer
        self.w_optim = torch.optim.AdamW(weights(self.model), self.w_lr,
                                         weight_decay=self.args.w_weight_decay)
        if if_print_w:
            print("="*20+"optimizer_weight"+"="*20)
            for n, p in named_weights(self.model):
                print(n)
            print("="*25+"="*25)
        # alphas optimizer
        if not self.args.for_baseline:
            self.alpha_optim = torch.optim.AdamW(
                alphas(self.model), self.alpha_lr, weight_decay=0.0)
            if if_print_alphas:
                print("="*20+"optimizer_alphas"+"="*20)
                for n, p in named_alphas(self.model):
                    print(n)
                print("="*25+"="*25)

    def set_architect(self):
        self.architect = Architect(self.model, self.v_model, self.delta_model, self.v_delta_model, self.args)

    def set_schedule(self):
        self.w_schedule = get_linear_schedule_with_warmup(
            self.w_optim, self.w_warmup_steps, self.max_step)
        if not self.args.for_baseline:
            self.alpha_schedule = get_linear_schedule_with_warmup(
                self.alpha_optim, self.alpha_warmup_steps, self.max_step)

    def train(self):

        logger.info(f"begin training")
        logger.info(f"max step: {self.max_step}")

        self.best_score = 0
        self.best_step = 0

        for _ in range(100000):
            if not self.args.for_baseline and self.args.using_darts:
                dataloader = zip(self.train_dataloader, self.eval_dataloader)
            else:
                dataloader = self.train_dataloader

            for step, inputs in enumerate(tqdm(dataloader)):

                if not self.args.for_baseline and self.args.using_darts:
                    trn_input, val_input = inputs
                else:
                    trn_input, val_input = inputs, None

                # one step
                self.train_step(trn_input, val_input)


                # do eval
                if self.global_step % self.valid_step == 0:
                    metrics = self.evaluate(self.eval_dataloader_not_shuffle, split='eval')
                    score = list(metrics.values())[0]
                    if score >= self.best_score:
                        self.best_score = score
                        self.best_step = self.global_step
                        if not os.path.exists(self.args.output_dir):
                            os.mkdir(self.args.output_dir)
                        torch.save(self.model.state_dict(), os.path.join(
                            self.args.output_dir, 'best.ckpt'))
                    logger.info(
                        f'step: {self.global_step}/{self.max_step}, metrics: {metrics}, best: {self.best_score}, best step: {self.best_step}')

                # print alphas
                if self.global_step % self.valid_step == 0 and not self.args.for_baseline:
                    to_print = '\n'
                    to_print += '='*25 + \
                        f'{self.global_step}/{self.max_step}'+'='*25+'\n'
                    p_alphas = print_alphas(self.delta_model)
                    to_print += p_alphas+'\n'
                    to_print += f'step: {self.global_step}/{self.max_step}'+'\n'
                    to_print += f'w_lr: {self.w_schedule.get_last_lr()}\nalpha_lr: {self.alpha_schedule.get_last_lr()}'+'\n'
                    to_print += f'best score: {self.best_score}'+'\n'
                    to_print += f'best step: {self.best_step}'+'\n'
                    to_print += '='*55+'\n'

                    logger.info(to_print)

                if self.args.lr_decay:
                    self.w_schedule.step()
                    if not self.args.for_baseline:
                        self.alpha_schedule.step()
                
                if self.args.sparsity_decay_step > 0 and not self.args.for_baseline:
                    self.sparsity_rate_schedule.step()

                if self.global_step >= self.max_step:
                    break

            if self.global_step >= self.max_step:
                break

    def train_step(self, trn_input, val_input):

        self.global_step += 1
        self.model.train()
        if self.v_model is not None:
            self.v_model.train()

        # copy model and do sampling
        if not self.args.for_baseline:
            if self.args.using_darts:
                self.architect.copy_params()
            if self.args.sample_once_per_step:
                # keep the sample the same in 4 loss
                us = torch.rand(len(self.delta_model.new_module_params))
                self.delta_model.do_sampling(us=us,random=True,hard_forward=self.args.hard_forward,training=True,using_last_us=False)
                if self.args.using_darts:
                    self.v_delta_model.do_sampling(us=us,random=True,hard_forward=self.args.hard_forward,training=True,using_last_us=False)

        trn_input['decoder_input_ids'] = self.model._shift_right(trn_input['labels'])
        for k, v in trn_input.items():
            trn_input[k] = v.to(self.model.device)

        if not self.args.for_baseline and self.args.using_darts:
            # optimize alphas
            val_input['decoder_input_ids'] = self.v_model._shift_right(
                val_input['labels'])
            for k, v in val_input.items():
                val_input[k] = v.to(self.v_model.device)

            self.alpha_optim.zero_grad()
            self.architect.unrolled_backward(
                trn_input, val_input, self.w_optim)

            torch.nn.utils.clip_grad_norm_(
                alphas(self.model), self.args.clip_grad_norm)
            self.alpha_optim.step()

        self.w_optim.zero_grad()
        # optimize w
        if not self.args.for_baseline and not self.args.using_darts:
            self.alpha_optim.zero_grad()
            self.delta_model.do_sampling(us=None,random=True,hard_forward=self.args.hard_forward,training=True,using_last_us=False)
            loss = self.architect.get_loss(
                True, trn_input, if_L0_norm=(self.args.global_strategy == "l0-norm"))
        else:
            if not self.args.for_baseline:
                self.delta_model.do_sampling(us=None,random=True,hard_forward=self.args.hard_forward,training=True,using_last_us=False)
            loss = self.architect.get_loss(
                True, trn_input, if_L0_norm=False)

        if loss.grad_fn is not None:
            loss.backward()
            if not self.args.for_baseline and not self.args.using_darts:
                torch.nn.utils.clip_grad_norm_(
                    alphas(self.model), self.args.clip_grad_norm)
                self.alpha_optim.step()

            torch.nn.utils.clip_grad_norm_(
                        weights(self.model), self.args.clip_grad_norm)
            self.w_optim.step()
        else:
            logger.info('loss has no grad.')
            logger.info(loss)
            
        return loss

    def load_ckpt(self, PATH=None):
        if PATH == None:
            PATH = os.path.join(self.args.output_dir, 'best.ckpt')
        assert os.path.exists(PATH), f"{PATH} not exists"
        pretrained_dict = torch.load(PATH)
        logger.info(f'load best ckpt from: {PATH}')
        self.model.load_state_dict(pretrained_dict, strict=False)
        if not self.args.for_baseline:
            self.delta_model.do_sampling(us=None,random=False,hard_forward=self.args.hard_forward,training=False,using_last_us=False)
            p_alphas = print_alphas(self.delta_model)
            logger.info(p_alphas)

    @torch.no_grad()
    def evaluate(self, dataloader, max_length=None, num_beams=None, split: str = None, PATH=None):
        logger.info('begin eval')
        if split == 'test':
            self.load_ckpt(PATH)

        if self.data_info is not None:
            assert split in ['eval', 'test']
            info = self.data_info[split]
        else:
            info = None

        if not self.args.for_baseline :
            self.delta_model.do_sampling(us=None,random=False,hard_forward=self.args.hard_forward,training=False,using_last_us=False)
            if self.args.hard_forward:
                params = sum(i['num_param'] for i in self.delta_model.new_module_params if i['z_delta']==1.0)
                logger.info(f'param: {params}')

        self.model.eval()
        self._max_length = max_length
        self._num_beams = num_beams
        loss_list = []
        outputs = []
        labels = []

        for inputs in tqdm(dataloader):
            loss, generated_tokens, label = self.prediction_step(inputs)
            loss_list.append(loss.item())
            outputs.append(generated_tokens.cpu())
            labels.append(label.cpu())


        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)
        
        # print(outputs,labels,info)
        result = self.compute_metrics((outputs, labels, info))
        logger.info(f'metrics: {result}')
        return result

    def prediction_step(
        self,
        inputs,
        prediction_loss_only: bool = False,
    ):

        for k, v in inputs.items():
            inputs[k] = v.to(self.model.device)
        has_labels = "labels" in inputs
        # inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": inputs["labels"].shape[-1]+10 if self.args.task_name=='web_nlg' else self.model.config.max_length,
            "num_beams": 5 if self.args.task_name=='web_nlg' else self.model.config.num_beams,
        }
        all_max_length = 192 if self.args.task_name=='web_nlg' else self.model.config.max_length

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        ).cpu()
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < all_max_length:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, all_max_length)

        loss = torch.Tensor([0])

        if prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"].cpu()
        if labels.shape[-1] < all_max_length:
            labels = self._pad_tensors_to_max_len(labels, all_max_length)

        if self.args.task_name in ["superglue-record"] and labels.shape[-1] > all_max_length:
            labels = labels[...,:all_max_length]

        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError(
                    "Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0],
             max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
