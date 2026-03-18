from pathlib import Path  
p = Path(r'D:\research\MARL_GIT\MARLlib\marllib\marl\algos\core\IL\ppo_extra_adam.py')  
s = p.read_text()  
old = '''        loss_2 = ppo_surrogate_loss(self, self.model, self.dist_class, loss_inputs)n        loss_2.backward()nn        grad_clip = self.config.get(\"grad_clip\", None)n'''  
new = '''        loss_2 = ppo_surrogate_loss(self, self.model, self.dist_class, loss_inputs)n        loss_2.backward()nn        mean_kl = self.model.tower_stats.get(\"mean_kl_loss\", torch.tensor(0.0))n        if hasattr(mean_kl, \"item\"):n            mean_kl = mean_kl.item()nn        grad_clip = self.config.get(\"grad_clip\", None)n'''  
if old not in s:  
    raise SystemExit('target block not found')  
p.write_text(s.replace(old, new))  
