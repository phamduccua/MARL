import copy
import torch

from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, ppo_surrogate_loss
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG

from marllib.marl.algos.utils.extra_adam import ExtraAdam
from marllib.marl.algos.utils.vi import apply_vi


IPPO_VI_CONFIG = copy.deepcopy(PPO_CONFIG)
IPPO_VI_CONFIG.update({
    "entropy_coeff_schedule": [
        [0, 0.02],        # start: explore nhiều
        [50000, 0.01],    # giảm dần
        [100000, 0.001],  # gần cuối: exploit
    ],
    "tau_vi": 0.05,
    "anchor_update_freq": 10,
    "vi_lr": None,              # nếu None thì dùng config["lr"]
    "vi_betas": (0.9, 0.999),
    "vi_eps": 1e-5,
    "vi_weight_decay": 0.0,
})


class IPPOVITorchPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        lr = config["vi_lr"] if config.get("vi_lr") is not None else config["lr"]

        self.vi_optimizer = ExtraAdam(
            self.model.parameters(),
            lr=lr,
            betas=tuple(config.get("vi_betas", (0.9, 0.999))),
            eps=config.get("vi_eps", 1e-5),
            weight_decay=config.get("vi_weight_decay", 0.0),
        )

        self.tau_vi = config.get("tau_vi", 0.05)
        self.anchor_update_freq = config.get("anchor_update_freq", 10)
        self.vi_update_count = 0
        self.anchor_params = [p.detach().clone() for p in self.model.parameters()]

    def _refresh_anchor_if_needed(self):
        self.vi_update_count += 1
        if self.vi_update_count % self.anchor_update_freq == 0:
            self.anchor_params = [p.detach().clone() for p in self.model.parameters()]

    def learn_on_batch(self, samples):
        """
        Ghi chú:
        - Đây là skeleton để cấy VI vào đúng chỗ update.
        - Nếu RLlib version của bạn không đi qua learn_on_batch theo cách này,
          cần dời logic xuống hàm SGD/minibatch đang được policy thực sự gọi.
        """

        self.model.train()

        # ----- Step 1: grad tại theta_t -----
        self.vi_optimizer.zero_grad(set_to_none=True)

        loss_inputs = self._lazy_tensor_dict(samples)
        loss_1 = ppo_surrogate_loss(self, self.model, self.dist_class, loss_inputs)
        loss_1.backward()

        self.vi_optimizer.extrapolate()

        # ----- Step 2: grad tại theta_tilde -----
        self.vi_optimizer.zero_grad(set_to_none=True)

        loss_inputs = self._lazy_tensor_dict(samples)
        loss_2 = ppo_surrogate_loss(self, self.model, self.dist_class, loss_inputs)
        loss_2.backward()

        apply_vi(self.model, self.anchor_params, self.tau_vi)

        mean_kl = self.model.tower_stats.get("mean_kl_loss", torch.tensor(0.0))
        if hasattr(mean_kl, "item"):
            mean_kl = mean_kl.item()

        grad_clip = self.config.get("grad_clip", None)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        self.vi_optimizer.restore()
        self.vi_optimizer.step()
        self.vi_optimizer.zero_grad(set_to_none=True)

        self._refresh_anchor_if_needed()

        return {
            "learner_stats": {
                "total_loss": float(loss_2.item()),
                "kl": float(mean_kl),
                "tau_vi": float(self.tau_vi),
            }
        }


def get_policy_class_ippo_vi(config_):
    if config_["framework"] == "torch":
        return IPPOVITorchPolicy
    return None


IPPOVITrainer = PPOTrainer.with_updates(
    name="IPPOVITrainer",
    default_policy=None,
    get_policy_class=get_policy_class_ippo_vi,
    default_config=IPPO_VI_CONFIG,
)