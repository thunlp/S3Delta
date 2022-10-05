import copy
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
import torch.nn.functional as F
import torch.optim._functional as optim_F
from s3delta import S3Delta, named_weights, weights, alphas, named_alphas, print_alphas
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_num_trainable_parameters(module: nn.Module):
    pnum_tot = 0
    for param in module.parameters():
        if param.requires_grad:
            pnum_tot += param.numel()
    return pnum_tot


class Architect():
    def __init__(self,
                 net: T5ForConditionalGeneration,
                 v_net: T5ForConditionalGeneration,
                 delta_model: S3Delta,
                 v_delta_model: S3Delta,
                 args):

        self.net = net
        self.delta_model = delta_model
        self.v_net = v_net
        self.v_delta_model = v_delta_model
        self.args = args

    def get_loss(self, net_type: bool, inputs, if_L0_norm=False):

        if net_type:
            model = self.net
            delta_model = self.delta_model
        else:
            model = self.v_net
            delta_model = self.v_delta_model

        outputs = model(**inputs)
        loss = outputs.loss

        # calculate L0 norm for alphas
        if if_L0_norm:
            assert self.args.r != 0
            L0_norm = 0
            bias = self.args.beta * torch.log(torch.tensor(-self.args.l/self.args.r))
            lambda0 = self.args.lambda0
            for idx, param in enumerate(delta_model.new_module_params):
                p_delta = param['p_delta']
                L0_norm += torch.sigmoid(torch.log(p_delta /
                                         (1-p_delta))-bias.to(p_delta.device))

            loss = lambda0 * L0_norm.to(loss.device) + loss

        return loss

    def copy_params(self):
        '''Explicitly copy the model: net -> v_net'''
        if self.v_net is None:
            return
        with torch.no_grad():
            for w, vw in zip(weights(self.net), weights(self.v_net)):
                vw.copy_(w)
            for a, va in zip(alphas(self.net), alphas(self.v_net)):
                va.copy_(a)

    def virtual_step(self, inputs, w_optim: torch.optim.Optimizer):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        self.delta_model.do_sampling(us=None, random=True, hard_forward=self.args.hard_forward,
                                     training=True, using_last_us=self.args.sample_once_per_step)
        loss = self.get_loss(True, inputs)  # L_trn(w)

        # compute gradient

        gradients = torch.autograd.grad(
            loss, weights(self.net), allow_unused=True)

        # do virtual step (update gradient)
        # Referenced from https://pytorch.org/docs/stable/_modules/torch/optim/adamw.html#AdamW
        with torch.no_grad():
            if isinstance(w_optim, torch.optim.AdamW):
                for group in w_optim.param_groups:
                    params_with_grad = []
                    grads = []
                    exp_avgs = []
                    exp_avg_sqs = []
                    max_exp_avg_sqs = []
                    state_steps = []
                    amsgrad = group['amsgrad']
                    beta1, beta2 = group['betas']

                    params_ids = [id(p) for p in group['params']]

                    # dict key is not the value, but the pointer. So original network weight have to
                    # be iterated also.
                    for w, vw, g in zip(weights(self.net), weights(self.v_net), gradients):
                        if id(w) in params_ids and g is not None:
                            vw.copy_(w)
                            params_with_grad.append(vw)
                            grads.append(g)
                            # use deepcopy here, cause adamw could modify exp_avgs & exp_avg_sqs
                            state = copy.deepcopy(w_optim.state[w])
                            if len(state) == 0:
                                state['step'] = torch.tensor(0.)
                                # Exponential moving average of gradient values
                                state['exp_avg'] = torch.zeros_like(
                                    vw, memory_format=torch.preserve_format)
                                # Exponential moving average of squared gradient values
                                state['exp_avg_sq'] = torch.zeros_like(
                                    vw, memory_format=torch.preserve_format)
                                if amsgrad:
                                    # Maintains max of all exp. moving avg. of sq. grad. values
                                    state['max_exp_avg_sq'] = torch.zeros_like(
                                        vw, memory_format=torch.preserve_format)

                            exp_avgs.append(state['exp_avg'])
                            exp_avg_sqs.append(state['exp_avg_sq'])
                            if amsgrad:
                                max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                            # state['step'] += 1
                            state_steps.append(state['step'])
                    optim_F.adamw(params_with_grad,
                                  grads,
                                  exp_avgs,
                                  exp_avg_sqs,
                                  max_exp_avg_sqs,
                                  state_steps,
                                  amsgrad=amsgrad,
                                  beta1=beta1,
                                  beta2=beta2,
                                  lr=group['lr'],
                                  weight_decay=group['weight_decay'],
                                  eps=group['eps'],
                                  maximize=group['maximize'])
            else:
                # adafactor need to be implemented 
                raise NotImplementedError

            # synchronize alphas
            for a, va in zip(alphas(self.net), alphas(self.v_net)):
                va.copy_(a)

    def unrolled_backward(self, trn_inputs, val_inputs, w_optim):
        """ Compute unrolled loss and backward its gradients

        """
        # do virtual step (calc w`)
        self.virtual_step(trn_inputs, w_optim)

        # calc unrolled loss
        self.v_delta_model.do_sampling(us=None, random=True, hard_forward=self.args.hard_forward,
                                       training=True, using_last_us=self.args.sample_once_per_step)
        v_loss = self.get_loss(False, val_inputs,
                               if_L0_norm=(self.args.global_strategy == "l0-norm"))  # L_val(w`)

        # compute gradient
        v_alphas = tuple(alphas(self.v_net))
        v_weights = tuple(weights(self.v_net))
        v_grads = torch.autograd.grad(
            v_loss, v_alphas + v_weights, allow_unused=True)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, trn_inputs)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            all_alphas = alphas(self.net)

            for alpha, da, h in zip(all_alphas, dalpha, hessian):
                alpha.grad = da - w_optim.param_groups[0]['lr']*h


    def compute_hessian(self, dw, trn_inputs):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw if w is not None]).norm()
        if norm == 0:
            print('norm is zero')
        eps = 0.01 / (norm+1e-6)

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(weights(self.net), dw):
                if d is not None:
                    p += eps * d

        self.delta_model.do_sampling(us=None, random=True, hard_forward=self.args.hard_forward,
                                     training=True, using_last_us=self.args.sample_once_per_step)
        loss1 = self.get_loss(True, trn_inputs, if_L0_norm=(
            self.args.global_strategy == "l0-norm"))

        dalpha_pos = torch.autograd.grad(
            loss1, alphas(self.net))  # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(weights(self.net), dw):
                if d is not None:
                    p -= 2. * eps * d
        self.delta_model.do_sampling(us=None, random=True, hard_forward=self.args.hard_forward,
                                     training=True, using_last_us=self.args.sample_once_per_step)
        loss2 = self.get_loss(True, trn_inputs, if_L0_norm=(self.args.global_strategy == "l0-norm"))

        dalpha_neg = torch.autograd.grad(
            loss2, alphas(self.net))  # dalpha { L_trn(w-) }


        # recover w
        with torch.no_grad():
            for p, d in zip(weights(self.net), dw):
                if d is not None:
                    p += eps * d

        hessian = [(p-n) / (2.*eps) for p, n in zip(dalpha_pos, dalpha_neg)]

        return hessian
