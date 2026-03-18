from marllib import marl

env = marl.make_env(
    environment_name="mpe",
    map_name="simple_spread",
    force_coop=True
)

algo = marl.algos.mappo(hyperparam_source="test")

model = marl.build_model(
    env,
    algo,
    {
        "core_arch": "mlp",
        "encode_layer": "128-128"
    }
)

algo.fit(
    env,
    model,
    stop={"training_iteration": 2},
    local_mode=True,
    num_gpus=0,
    num_workers=0,
    share_policy="group",
    checkpoint_freq=1
)