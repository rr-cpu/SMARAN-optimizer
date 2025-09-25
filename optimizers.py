import torch
import numpy as np
import torch.optim as optim
import math
import logging
import os
import torch.distributed as dist
from typing import TYPE_CHECKING, Any, Callable, Optional
import time
import copy
import contextlib
if TYPE_CHECKING:
    from optim.optimizer import _params_t
else:
    _params_t = Any
    

#Decomposed gradient descent
class DecGD(optim.Optimizer):
    def __init__(self, params, lr=0.01, c=1, gamma=0.1, ams=False, device=torch.device("cpu")):
        defaults = dict(lr=lr, c=c, gamma=gamma, ams=ams)
        super(DecGD, self).__init__(params, defaults)
        self.device=device

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                gamma = group["gamma"]
                c = group["c"]
                lr = group["lr"]
                ams = group["ams"]

                # Initialize parameters for first iteration
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.data, device=self.device)
                    state["v"] = torch.ones_like(p.data, device=self.device) * (torch.sqrt(torch.tensor(loss.item(), device=self.device) + c))
                    state["previous_p"] = torch.zeros_like(p.data, device=self.device)
                    state['v_star'] = torch.full_like(state["v"], float('inf'), device=self.device)

                m = state["m"]
                v = state["v"]
                prev_p = state['previous_p']  # Retrieve previous value
                v_star = state['v_star']      # Retrieve stored v*
                

                # Compute scaled gradient
                delta_g = grad / (2 * torch.sqrt(torch.tensor(loss.item(), device=self.device) + c)) if loss else grad

                # Update momentum
                m.mul_(gamma).add_(delta_g)

                # Update v
                v.add_(m * (p.data - prev_p))

                # AMS modification
                if ams:
                    v_star = torch.min(v_star, v)
                else:
                    v_star = v

                # Update the state
                state["v_star"] = v_star  # Update stored v*
                state["m"] = m
                state["v"] = v
                state['previous_p'] = p.data.clone()  # Update previous value in state

                # Update parameters
                p.data.add_(-2 * lr * v_star * m)

        return loss



#SMARAN
class SMARAN(optim.Optimizer):
    def __init__(self, params, lr=0.01, gamma=0.1, weight_decay=0.01, device=torch.device("cpu")):
        defaults = dict(lr=lr, gamma=gamma)
        super(SMARAN, self).__init__(params, defaults)
        self.T=0
        self.device=device
        self.weight_decay=weight_decay

    def step(self, closure=None):
        self.T=self.T+1
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                # Initialize parameters for first iteration
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.data, device=self.device)
                    state['v']=0

                m = state["m"]
                v = state["v"]
                gamma = group["gamma"]
                lr = group["lr"]
                #normalized gradient
                delta_g = grad/(torch.norm(grad,p=2)+ 1e-8)

                
                # Update momentum
                m.mul_(gamma).add_(delta_g)

                #second moment of loss
                v= gamma * v + loss.item()**2

                # Update the state
                state["m"] = m
                state["v"] = v
                #update parameters
                # p.data.add_(-1*lr*(f1 + loss.item())*(m + self.weight_decay*p.data)/(np.sqrt(v) +1e-8))
                p.data.add_(-1*lr*loss.item()*(m + self.weight_decay*p.data)/(np.sqrt(v) +1e-8))

        return loss



#prodigy implementation from official github repo https://github.com/konstmish/prodigy/blob/main/prodigyopt/prodigy.py
class Prodigy(optim.Optimizer):
    r"""
    Implements Adam with Prodigy step-sizes.
    Leave LR set to 1 unless you encounter instability.
   
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate adjustment parameter. Increases or decreases the Prodigy learning rate.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        beta3 (float):
            coefficients for computing the Prodidy stepsize using running averages.
            If set to None, uses the value of square root of beta2 (default: None).
        eps (float):
            Term added to the denominator outside of the root operation to improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        decouple (boolean):
            Use AdamW style decoupled weight decay
        use_bias_correction (boolean):
            Turn on Adam's bias correction. Off by default.
        safeguard_warmup (boolean):
            Remove lr from the denominator of D estimate to avoid issues during warm-up stage. Off by default.
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        d_coef (float):
            Coefficient in the expression for the estimate of d (default 1.0).
            Values such as 0.5 and 2.0 typically work as well. 
            Changing this parameter is the preferred way to tune the method.
        growth_rate (float):
            prevent the D estimate from growing faster than this multiplicative rate.
            Default is inf, for unrestricted. Values like 1.02 give a kind of learning
            rate warmup effect.
        fsdp_in_use (bool):
            If you're using sharded parameters, this should be set to True. The optimizer
            will attempt to auto-detect this, but if you're using an implementation other
            than PyTorch's builtin version, the auto-detection won't work.
        slice_p (int): Reduce memory usage by calculating LR adaptation statistics on only every 
            pth entry of each tensor. For values greater than 1 this is an approximation to standard 
            Prodigy. Values ~11 are reasonable (default 1).
    """
    def __init__(self, params, lr=1.0,
                 betas=(0.9, 0.999), beta3=None,
                 eps=1e-8, weight_decay=0, decouple=True, 
                 use_bias_correction=False, safeguard_warmup=False,
                 d0=1e-6, d_coef=1.0, growth_rate=float('inf'),
                 fsdp_in_use=False,
                 slice_p=1):
        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        if decouple and weight_decay > 0:
            print(f"Using decoupled weight decay")

       
        defaults = dict(lr=lr, betas=betas, beta3=beta3,
                        eps=eps, weight_decay=weight_decay,
                        d=d0, d0=d0, d_max=d0,
                        d_numerator=0.0, d_coef=d_coef,
                        k=0, growth_rate=growth_rate,
                        use_bias_correction=use_bias_correction,
                        decouple=decouple, safeguard_warmup=safeguard_warmup,
                        fsdp_in_use=fsdp_in_use,
                        slice_p=slice_p)
        self.d0 = d0
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        d_denom = 0.0

        group = self.param_groups[0]
        use_bias_correction = group['use_bias_correction']
        beta1, beta2 = group['betas']
        beta3 = group['beta3']
        if beta3 is None:
            beta3 = math.sqrt(beta2)
        k = group['k']

        d = group['d']
        d_max = group['d_max']
        d_coef = group['d_coef']
        lr = max(group['lr'] for group in self.param_groups)

        if use_bias_correction:
            bias_correction = ((1 - beta2**(k+1))**0.5) / (1 - beta1**(k+1))
        else:
            bias_correction = 1

        dlr = d*lr*bias_correction
       
        growth_rate = group['growth_rate']
        decouple = group['decouple']
        fsdp_in_use = group['fsdp_in_use']

        d_numerator = group['d_numerator']
        d_numerator *= beta3
        delta_numerator = 0.0

        for group in self.param_groups:
            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']
            group_lr = group['lr']
            d0 = group['d0']
            safeguard_warmup = group['safeguard_warmup']
            slice_p = group['slice_p']

            if group_lr not in [lr, 0.0]:
                raise RuntimeError(f"Setting different lr values in different parameter groups is only supported for values of 0")

            for p in group['params']:
                if p.grad is None:
                    continue
                if hasattr(p, "_fsdp_flattened"):
                    fsdp_in_use = True
               
                grad = p.grad.data
               
                # Apply weight decay (coupled variant)
                if decay != 0 and not decouple:
                    grad.add_(p.data, alpha=decay)

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0

                    state['s'] = torch.zeros_like(p.data.flatten()[::slice_p]).detach()

                    if p.any():
                        state['p0'] = p.flatten()[::slice_p].detach().clone()
                    else:
                        # All values are zero, so save VRAM with a zero-tensor
                        state['p0'] = torch.tensor(0, device=p.device, dtype=p.dtype)

                    # Exponential moving average of gradient values
                    if beta1 > 0:
                        state['exp_avg'] = torch.zeros_like(p.data).detach()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data).detach()

                exp_avg_sq = state['exp_avg_sq']

                s = state['s']
                p0 = state['p0']

                if group_lr > 0.0:
                    # we use d / d0 instead of just d to avoid getting values that are too small
                    sliced_grad = grad.flatten()[::slice_p]
                    delta_numerator += (d / d0) * dlr * torch.dot(sliced_grad, p0.data - p.data.flatten()[::slice_p]).item()

                    # Adam EMA updates
                    if beta1 > 0:
                        exp_avg = state['exp_avg']
                        exp_avg.mul_(beta1).add_(grad, alpha=d * (1-beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=d * d * (1-beta2))

                    if safeguard_warmup:
                        s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * d))
                    else:
                        s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * dlr))
                    d_denom += s.abs().sum().item()

            ######

        d_hat = d

        # if we have not done any progres, return
        # if we have any gradients available, will have d_denom > 0 (unless \|g\|=0)
        if d_denom == 0 and not fsdp_in_use:
            return loss
       
        if lr > 0.0:
            if fsdp_in_use:
                dist_tensor = torch.zeros(2).cuda()
                dist_tensor[0] = delta_numerator
                dist_tensor[1] = d_denom
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_d_numerator = d_numerator + dist_tensor[0]
                global_d_denom = dist_tensor[1]
            else:
                global_d_numerator = d_numerator + delta_numerator
                global_d_denom = d_denom

            d_hat = d_coef * global_d_numerator / global_d_denom
            if d == group['d0']:
                d = max(d, d_hat)
            d_max = max(d_max, d_hat)
            d = min(d_max, d * growth_rate)

        for group in self.param_groups:
            group['d_numerator'] = global_d_numerator
            group['d_denom'] = global_d_denom
            group['d'] = d
            group['d_max'] = d_max
            group['d_hat'] = d_hat

            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                exp_avg_sq = state['exp_avg_sq']

                state['step'] += 1

                denom = exp_avg_sq.sqrt().add_(d * eps)

                # Apply weight decay (decoupled variant)
                if decay != 0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)

                ### Take step
                if beta1 > 0:
                    exp_avg = state['exp_avg']
                    p.data.addcdiv_(exp_avg, denom, value=-dlr)
                else:
                    p.data.addcdiv_(grad, denom, value=-dlr * d)

            group['k'] = k + 1

        return loss





class PoNoS(torch.optim.Optimizer):    # Taken from https://github.com/mconnx21/AdaPoNoS/blob/master/PoNoS.py
    """
    PoNoS Arguments:
         c=0.5, # line search sufficient decrease scaling constant
         c_p=0.1, # Polyak step size scaling constant
         delta=0.5, # cutting step
         zhang_xi=1, # Zhang xi, controlling the nonmonotonicity
         max_eta=10, # maximum step size
         min_eta=1e-06, #minimum step size
         f_star=0, # estimate of the min value of f
         save_backtracks=True # activate the memory-based resetting technique

         Note that PoNoS is like LBFGS from the LBFGS optimizer from pytorch,
         the step needs to be called like in the following:
         closure = lambda: loss_function(model, images, labels, backwards=False)
         opt.step(closure)
    """

    def __init__(self,
                 params,
                 c=0.5,
                 c_p=0.1,
                 delta=0.5,
                 zhang_xi=1,
                 max_eta=10,
                 min_eta=1e-06,
                 f_star=0,
                 save_backtracks=True):

        params = list(params)
        super().__init__(params, {})

        self.params = params
        self.c = c
        self.delta = delta
        self.lk = 0
        self.zhang_xi = zhang_xi

        self.state["Q_k"] = 0
        self.state["C_k"] = 0
        self.max_eta = max_eta
        self.min_eta = min_eta
        self.c_p = c_p
        self.save_backtracks = save_backtracks
        self.f_star = f_star

    def step(self, closure):
        # deterministic closure
        seed = time.time()
        def closure_deterministic():
            with self.random_seed_torch(int(seed)):
                return closure()

        # get loss and compute gradients
        loss = closure_deterministic()
        loss.backward()

        # save the current parameters:
        params_current = copy.deepcopy(self.params)
        grad_current = self.get_grad_list(self.params)
        grad_norm = self.compute_grad_norm(grad_current)

        # setting the Polyak initial step size
        polyak_step_size = loss / (self.c_p * grad_norm ** 2 + 1e-8)
        if self.save_backtracks:
            polyak_step_size = polyak_step_size * (self.delta ** self.lk)
        step_size = max(min(polyak_step_size, self.max_eta), self.min_eta)

        self.line_search(step_size, params_current, grad_current, loss, closure_deterministic, grad_norm)
        return loss

    def line_search(self, step_size, params_current, grad_current, loss, closure_deterministic, grad_norm):
        with torch.no_grad():

            # compute nonmonotone terms for the Zhang&Hager line search
            q_kplus1 = self.zhang_xi * self.state["Q_k"] + 1
            self.state["C_k"] = (self.zhang_xi * self.state["Q_k"] * self.state["C_k"] + loss.item()) / q_kplus1
            self.state["Q_k"] = q_kplus1

            grad_norm = self.maybe_torch(grad_norm)
            if grad_norm >= 1e-8 and loss.item() != 0:
                # check if condition is satisfied
                found = 0

                suff_dec = grad_norm ** 2

                for e in range(100):
                    # try a prospective step
                    self.try_sgd_update(self.params, step_size, params_current, grad_current)

                    # compute the loss at the next step; no need to compute gradients.
                    loss_next = closure_deterministic()
                    ref_value = max(self.state["C_k"], loss.item())
                    found, step_size = self.check_armijo_conditions(step_size=step_size,
                                                                    loss=ref_value,
                                                                    suff_dec=suff_dec,
                                                                    loss_next=loss_next,
                                                                    c=self.c,
                                                                    beta_b=self.delta)

                    if found == 1:
                        break

                # if line search exceeds 100 internal iterations
                if found == 0:
                    step_size = torch.tensor(data=1e-6)
                    self.try_sgd_update(self.params, 1e-6, params_current, grad_current)

                self.lk = max(self.lk + e - 1, 0)

            else:
                print("Grad norm is {} and loss is {}".format(grad_norm, loss.item()))
                step_size = 0
                loss_next = closure_deterministic()

        return step_size, loss_next

    def maybe_torch(self,value):
        if isinstance(value, torch.Tensor):
            return value.item()
        else:
            return value

    # Armijo line search
    def check_armijo_conditions(self, step_size, loss, suff_dec,
                                loss_next, c, beta_b):
        found = 0
        sufficient_decrease = (step_size) * c * suff_dec
        rhs = loss - sufficient_decrease
        break_condition = loss_next - rhs
        if (break_condition <= 0):
            found = 1
        else:
            step_size = step_size * beta_b

        return found, step_size

    def try_sgd_update(self, params, step_size, params_current, grad_current):
        zipped = zip(params, params_current, grad_current)

        for p_next, p_current, g_current in zipped:
            p_next.data = p_current - step_size * g_current

    def compute_grad_norm(self, grad_list):
        grad_norm = 0.
        for g in grad_list:
            if g is None:
                continue
            grad_norm += torch.sum(torch.mul(g, g))
        grad_norm = torch.sqrt(grad_norm)
        return grad_norm

    @contextlib.contextmanager
    def random_seed_torch(self, seed, device=0):
        cpu_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            gpu_rng_state = torch.cuda.get_rng_state(0)

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        try:
            yield
        finally:
            torch.set_rng_state(cpu_rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(gpu_rng_state, device)

    def get_grad_list(self, params):
        return [p.grad for p in params]


        