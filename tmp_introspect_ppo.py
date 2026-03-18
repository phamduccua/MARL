import ray.rllib.agents.ppo.ppo_torch_policy as m 
print([n for n in dir(m) if 'loss' in n.lower()]) 
print([n for n in dir(m.PPOTorchPolicy) if 'loss' in n.lower()])
