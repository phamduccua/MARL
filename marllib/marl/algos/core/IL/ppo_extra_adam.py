import copy 
import torch 
 
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, ppo_surrogate_loss 
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG 
 
from marllib.marl.algos.utils.extra_adam import ExtraAdam 
 
 
IPPO_EXTRA_ADAM_CONFIG = copy.deepcopy(PPO_CONFIG) 
IPPO_EXTRA_ADAM_CONFIG.update({ 
    "extra_adam_lr": None, 
    "extra_adam_betas": (0.9, 0.999), 
    "extra_adam_eps": 1e-5, 
    "extra_adam_weight_decay": 0.0, 
}) 
 
 
class IPPOExtraAdamTorchPolicy(PPOTorchPolicy): 
    def __init__(self, observation_space, action_space, config): 
        super().__init__(observation_space, action_space, config) 
 
        lr = config["extra_adam_lr"] if config.get("extra_adam_lr") is not None else config["lr"] 
 
        self.extra_adam_optimizer = ExtraAdam( 
            self.model.parameters(), 
            lr=lr, 
            betas=tuple(config.get("extra_adam_betas", (0.9, 0.999))), 
            eps=config.get("extra_adam_eps", 1e-5), 
            weight_decay=config.get("extra_adam_weight_decay", 0.0), 
        ) 
 
    def learn_on_batch(self, samples): 
        self.model.train() 
 
        self.extra_adam_optimizer.zero_grad(set_to_none=True) 
        loss_inputs = self._lazy_tensor_dict(samples) 
        loss_1 = ppo_surrogate_loss(self, self.model, self.dist_class, loss_inputs) 
        loss_1.backward() 
 
        self.extra_adam_optimizer.extrapolate() 
 
        self.extra_adam_optimizer.zero_grad(set_to_none=True) 
        loss_inputs = self._lazy_tensor_dict(samples) 
        loss_2 = ppo_surrogate_loss(self, self.model, self.dist_class, loss_inputs) 
        loss_2.backward() 
 
        grad_clip = self.config.get("grad_clip", None) 
        if grad_clip is not None: 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip) 
 
        self.extra_adam_optimizer.restore() 
        self.extra_adam_optimizer.step() 
        self.extra_adam_optimizer.zero_grad(set_to_none=True) 
 
        return {"learner_stats": {"total_loss": float(loss_2.item()),
                "kl": float(mean_kl)}} 
 
 
def get_policy_class_ippo_extra_adam(config_): 
    if config_["framework"] == "torch": 
        return IPPOExtraAdamTorchPolicy 
    return None 
 
 
IPPOExtraAdamTrainer = PPOTrainer.with_updates( 
    name="IPPOExtraAdamTrainer", 
    default_policy=None, 
    get_policy_class=get_policy_class_ippo_extra_adam, 
    default_config=IPPO_EXTRA_ADAM_CONFIG, 
)
