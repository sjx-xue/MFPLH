import math

import torch
from torch.optim import Optimizer


class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch-1) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                           (1 + math.cos(math.pi * (epoch-1 - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr


class DenseSparseAdam(Optimizer):
    """
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(DenseSparseAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Parameters
        ----------
        closure : ``callable``, optional.
            A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                if 'exp_avg' not in state:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                if 'exp_avg_sq' not in state:
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'] += 1

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                weight_decay = group['weight_decay']

                if grad.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)

                    # Decay the first and second moment running average coefficient
                    #      old <- b * old + (1 - b) * new
                    # <==> old += (1 - b) * (new - old)
                    old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
                    exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
                    exp_avg.add_(make_sparse(exp_avg_update_values))
                    old_exp_avg_sq_values = exp_avg_sq.sparse_mask(grad)._values()
                    exp_avg_sq_update_values = grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
                    exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))

                    # Dense addition again is intended, avoiding another sparse_mask
                    numer = exp_avg_update_values.add_(old_exp_avg_values)
                    exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
                    denom = exp_avg_sq_update_values.sqrt_().add_(group['eps'])
                    del exp_avg_update_values, exp_avg_sq_update_values

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    p.data.add_(make_sparse(-step_size * numer.div_(denom)))
                    if weight_decay > 0.0:
                        p.data.add_(-group['lr'] * weight_decay, p.data.sparse_mask(grad))
                else:
                    # Decay the first and second moment running average coefficient
                    # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    # SJX
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    # SJX
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    p.data.addcdiv_(-step_size, exp_avg, denom)
                    if weight_decay > 0.0:
                        p.data.add_(-group['lr'] * weight_decay, p.data)

        return