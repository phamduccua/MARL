import torch


@torch.no_grad()
def apply_vi(model, anchor, tau):
    for p, a in zip(model.parameters(), anchor):
        if p.grad is None:
            continue
        p.grad.add_(p.data - a, alpha=tau)