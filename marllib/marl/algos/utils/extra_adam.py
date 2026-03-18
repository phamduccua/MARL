import math
import torch


class ExtraAdam:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-5, weight_decay=0.0):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.state = {}
        for p in self.params:
            self.state[p] = {
                "step": 0,
                "exp_avg": torch.zeros_like(p, memory_format=torch.preserve_format),
                "exp_avg_sq": torch.zeros_like(p, memory_format=torch.preserve_format),
            }

        self.param_backup = None

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    @torch.no_grad()
    def _compute_adam_update(self, p, grad, advance_state):
        if self.weight_decay != 0.0:
            grad = grad.add(p, alpha=self.weight_decay)

        state = self.state[p]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        step = state["step"] + 1

        new_exp_avg = exp_avg.mul(self.beta1).add(grad, alpha=1.0 - self.beta1)
        new_exp_avg_sq = exp_avg_sq.mul(self.beta2).addcmul(grad, grad, value=1.0 - self.beta2)

        bias_correction1 = 1.0 - self.beta1 ** step
        bias_correction2 = 1.0 - self.beta2 ** step

        denom = new_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
        denom = denom.add(self.eps)

        step_size = self.lr / bias_correction1
        update = new_exp_avg / denom
        update = update * step_size

        if advance_state:
            state["step"] = step
            state["exp_avg"].copy_(new_exp_avg)
            state["exp_avg_sq"].copy_(new_exp_avg_sq)

        return update

    @torch.no_grad()
    def extrapolate(self):
        if self.param_backup is not None:
            raise RuntimeError("Call restore() before calling extrapolate() again.")

        self.param_backup = [p.detach().clone() for p in self.params]

        for p in self.params:
            if p.grad is None:
                continue
            update = self._compute_adam_update(p, p.grad.detach(), advance_state=False)
            p.add_(-update)

    @torch.no_grad()
    def restore(self):
        if self.param_backup is None:
            return
        for p, backup in zip(self.params, self.param_backup):
            p.copy_(backup)
        self.param_backup = None

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            update = self._compute_adam_update(p, p.grad.detach(), advance_state=True)
            p.add_(-update)