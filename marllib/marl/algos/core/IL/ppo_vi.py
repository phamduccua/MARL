"""
ippo_vi_trainer.py

IPPO + VI (post-step interpolation) for RLlib PPO TorchPolicy.

Ý tưởng:
- Giữ extra-gradient style optimizer bằng ExtraAdam
- Không chèn VI vào giữa extrapolate / restore
- Apply VI sau optimizer.step()
- Refresh anchor theo tần suất cấu hình
- Có warmup cho tau_vi
- Có validate config, logging metric cơ bản, xử lý an toàn hơn

Lưu ý:
- File này giả định RLlib version của bạn vẫn dùng:
    ray.rllib.agents.ppo.ppo_torch_policy
    ray.rllib.agents.ppo.ppo
- Nếu bạn dùng RLlib version mới hơn, import path có thể đã đổi.
- learn_on_batch() có thể chưa phải hook minibatch tối ưu nhất ở mọi version RLlib.
"""

from __future__ import annotations

import copy
from typing import Dict, Any, List

import torch

from ray.rllib.agents.ppo.ppo_torch_policy import (
    PPOTorchPolicy,
    ppo_surrogate_loss,
)
from ray.rllib.agents.ppo.ppo import (
    PPOTrainer,
    DEFAULT_CONFIG as PPO_CONFIG,
)

from marllib.marl.algos.utils.extra_adam import ExtraAdam


def apply_vi(model: torch.nn.Module, anchor_params: List[torch.Tensor], tau_vi: float) -> None:
    """
    Post-step interpolation:
        theta <- (1 - tau) * theta + tau * theta_anchor

    Args:
        model: policy model hiện tại
        anchor_params: snapshot params làm anchor
        tau_vi: hệ số kéo về anchor, trong [0, 1]
    """
    if not 0.0 <= tau_vi <= 1.0:
        raise ValueError(f"tau_vi must be in [0, 1], got {tau_vi}")

    with torch.no_grad():
        model_params = list(model.parameters())
        if len(model_params) != len(anchor_params):
            raise ValueError(
                "anchor_params length does not match model.parameters(). "
                f"{len(anchor_params)} != {len(model_params)}"
            )

        for param, anchor in zip(model_params, anchor_params):
            if param.shape != anchor.shape:
                raise ValueError(
                    "anchor param shape mismatch: "
                    f"{tuple(param.shape)} != {tuple(anchor.shape)}"
                )
            param.data.mul_(1.0 - tau_vi).add_(anchor.data, alpha=tau_vi)


IPPO_VI_CONFIG = copy.deepcopy(PPO_CONFIG)
IPPO_VI_CONFIG.update(
    {
        "tau_vi": 0.05,
        "anchor_update_freq": 10,
        "vi_lr": None,                # None -> dùng config["lr"]
        "vi_betas": (0.9, 0.999),
        "vi_eps": 1e-5,
        "vi_weight_decay": 0.0,
        "vi_warmup_steps": 0,         # 0 = disable warmup
        "vi_max_tau": None,           # None = không clamp thêm
        "enable_vi": True,
    }
)


class IPPOVITorchPolicy(PPOTorchPolicy):
    """
    PPO Torch policy + ExtraAdam + VI post-step interpolation.
    """

    def __init__(self, observation_space, action_space, config: Dict[str, Any]):
        super().__init__(observation_space, action_space, config)

        self._validate_vi_config(config)

        lr = config["vi_lr"] if config.get("vi_lr") is not None else config["lr"]

        self.vi_optimizer = ExtraAdam(
            self.model.parameters(),
            lr=lr,
            betas=tuple(config.get("vi_betas", (0.9, 0.999))),
            eps=float(config.get("vi_eps", 1e-5)),
            weight_decay=float(config.get("vi_weight_decay", 0.0)),
        )

        self.enable_vi = bool(config.get("enable_vi", True))
        self.tau_vi = float(config.get("tau_vi", 0.005))
        self.anchor_update_freq = int(config.get("anchor_update_freq", 2))
        self.vi_warmup_steps = int(config.get("vi_warmup_steps", 0))
        self.vi_max_tau = config.get("vi_max_tau", None)

        self.vi_update_count = 0
        self.global_vi_step = 0
        self.anchor_params = self._clone_anchor_params()

    @staticmethod
    def _validate_vi_config(config: Dict[str, Any]) -> None:
        tau_vi = float(config.get("tau_vi", 0.005))
        anchor_update_freq = int(config.get("anchor_update_freq", 2))
        vi_lr = config.get("vi_lr", None)
        vi_warmup_steps = int(config.get("vi_warmup_steps", 0))
        vi_eps = float(config.get("vi_eps", 1e-5))
        vi_weight_decay = float(config.get("vi_weight_decay", 0.0))
        enable_vi = bool(config.get("enable_vi", True))
        vi_max_tau = config.get("vi_max_tau", None)

        if not isinstance(enable_vi, bool):
            raise TypeError("enable_vi must be bool")

        if not 0.0 <= tau_vi <= 1.0:
            raise ValueError("tau_vi must be in [0, 1]")

        if anchor_update_freq < 1:
            raise ValueError("anchor_update_freq must be >= 1")

        if vi_lr is not None and float(vi_lr) <= 0.0:
            raise ValueError("vi_lr must be > 0 when provided")

        if vi_warmup_steps < 0:
            raise ValueError("vi_warmup_steps must be >= 0")

        if vi_eps <= 0.0:
            raise ValueError("vi_eps must be > 0")

        if vi_weight_decay < 0.0:
            raise ValueError("vi_weight_decay must be >= 0")

        if vi_max_tau is not None and not 0.0 <= float(vi_max_tau) <= 1.0:
            raise ValueError("vi_max_tau must be in [0, 1] when provided")

    @torch.no_grad()
    def _clone_anchor_params(self) -> List[torch.Tensor]:
        return [param.detach().clone() for param in self.model.parameters()]

    @torch.no_grad()
    def _refresh_anchor_if_needed(self) -> None:
        self.vi_update_count += 1
        if self.vi_update_count % self.anchor_update_freq == 0:
            self.anchor_params = self._clone_anchor_params()

    def _get_effective_tau(self) -> float:
        if not self.enable_vi:
            return 0.0

        self.global_vi_step += 1

        if self.vi_warmup_steps <= 0:
            tau = self.tau_vi
        else:
            warmup_ratio = min(1.0, self.global_vi_step / self.vi_warmup_steps)
            tau = self.tau_vi * warmup_ratio

        if self.vi_max_tau is not None:
            tau = min(tau, float(self.vi_max_tau))

        return float(tau)

    def _get_mean_kl(self) -> float:
        mean_kl = 0.0
        tower_stats = getattr(self.model, "tower_stats", None)

        if isinstance(tower_stats, dict):
            mean_kl = tower_stats.get("mean_kl_loss", 0.0)

        if hasattr(mean_kl, "item"):
            mean_kl = mean_kl.item()

        try:
            return float(mean_kl)
        except (TypeError, ValueError):
            return 0.0

    def _get_grad_norm(self) -> float:
        total_norm_sq = 0.0

        for param in self.model.parameters():
            if param.grad is None:
                continue

            grad_norm = param.grad.data.norm(2).item()
            total_norm_sq += grad_norm * grad_norm

        return total_norm_sq ** 0.5

    @torch.no_grad()
    def _distance_to_anchor(self) -> float:
        total_sq = 0.0

        for param, anchor in zip(self.model.parameters(), self.anchor_params):
            diff = param - anchor
            total_sq += torch.sum(diff * diff).item()

        return total_sq ** 0.5

    def _build_loss_inputs(self, samples):
        """
        Tách riêng để dễ thay đổi nếu RLlib version khác.
        """
        return self._lazy_tensor_dict(samples)

    def learn_on_batch(self, samples):
        """
        Flow:
        1. Tính grad tại theta_t
        2. extrapolate() -> theta_tilde
        3. Tính grad tại theta_tilde
        4. restore() về theta_t
        5. optimizer.step() dùng grad tại theta_tilde
        6. apply_vi() sau step
        7. refresh anchor nếu cần
        """
        self.model.train()

        effective_tau = self._get_effective_tau()

        # ----- Step 1: grad tại theta_t -----
        self.vi_optimizer.zero_grad(set_to_none=True)

        loss_inputs_1 = self._build_loss_inputs(samples)
        loss_1 = ppo_surrogate_loss(self, self.model, self.dist_class, loss_inputs_1)
        loss_1.backward()

        self.vi_optimizer.extrapolate()

        # ----- Step 2: grad tại theta_tilde -----
        self.vi_optimizer.zero_grad(set_to_none=True)

        loss_inputs_2 = self._build_loss_inputs(samples)
        loss_2 = ppo_surrogate_loss(self, self.model, self.dist_class, loss_inputs_2)
        loss_2.backward()

        grad_clip = self.config.get("grad_clip", None)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        grad_norm = self._get_grad_norm()
        mean_kl = self._get_mean_kl()

        # ----- Restore + actual optimizer step -----
        self.vi_optimizer.restore()
        self.vi_optimizer.step()

        # ----- VI post-step interpolation -----
        if effective_tau > 0.0:
            apply_vi(self.model, self.anchor_params, effective_tau)

        distance_to_anchor = self._distance_to_anchor()

        self.vi_optimizer.zero_grad(set_to_none=True)
        self._refresh_anchor_if_needed()

        return {
            "learner_stats": {
                "loss_1": float(loss_1.item()),
                "total_loss": float(loss_2.item()),
                "kl": float(mean_kl),
                "tau_vi": float(effective_tau),
                "grad_norm": float(grad_norm),
                "distance_to_anchor": float(distance_to_anchor),
                "enable_vi": float(1.0 if self.enable_vi else 0.0),
            }
        }


def get_policy_class_ippo_vi(config_: Dict[str, Any]):
    if config_.get("framework") == "torch":
        return IPPOVITorchPolicy
    return None


IPPOVITrainer = PPOTrainer.with_updates(
    name="IPPOVITrainer",
    default_policy=None,
    get_policy_class=get_policy_class_ippo_vi,
    default_config=IPPO_VI_CONFIG,
)


__all__ = [
    "apply_vi",
    "IPPO_VI_CONFIG",
    "IPPOVITorchPolicy",
    "IPPOVITrainer",
    "get_policy_class_ippo_vi",
]