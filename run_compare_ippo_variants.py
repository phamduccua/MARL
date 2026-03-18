 
import argparse 
import ast 
import json 
 
from marllib import marl 
 
DEFAULT_ALGOS = ["ippo", "ippo_vi", "ippo_extra_adam"] 
DEFAULT_MODEL_CONFIG = { 
    "core_arch": "mlp", 
    "encode_layer": "128-128", 
} 
DEFAULT_STOP = { 
    "training_iteration": 2, 
} 
 
def parse_scalar(value): 
    lower = value.lower() 
    if lower == "true": 
        return True 
    if lower == "false": 
        return False 
    if lower in {"none", "null"}: 
        return None 
    try: 
        return ast.literal_eval(value) 
    except (ValueError, SyntaxError): 
        return value 
 
def parse_kv_list(items): 
    parsed = {} 
    for item in items or []: 
        if "=" not in item: 
            raise ValueError(f"Expected key=value, got: {item}") 
        key, value = item.split("=", 1) 
        parsed[key] = parse_scalar(value) 
    return parsed

def make_env(environment_name, map_name, force_coop, extra_env_args): 
    env_kwargs = { 
        "environment_name": environment_name, 
        "map_name": map_name, 
    } 
    if force_coop is not None: 
        env_kwargs["force_coop"] = force_coop 
    env_kwargs.update(extra_env_args) 
    return marl.make_env(**env_kwargs) 
 
def summarize_result(result): 
    if isinstance(result, dict): 
        last_result = result 
    elif hasattr(result, "trials") and result.trials: 
        last_result = result.trials[0].last_result or {} 
    else: 
        return {"result_type": type(result).__name__} 
 
    return { 
        "episode_reward_mean": last_result.get("episode_reward_mean"), 
        "timesteps_total": last_result.get("timesteps_total"), 
        "training_iteration": last_result.get("training_iteration"), 
        "time_total_s": last_result.get("time_total_s"), 
    } 
 
def run_once(algo_name, args, stop, model_config, fit_kwargs, algo_overrides, extra_env_args): 
    env = make_env(args.environment_name, args.map_name, args.force_coop, extra_env_args) 
    algo = getattr(marl.algos, algo_name)(hyperparam_source=args.hyperparam_source, **algo_overrides) 
    model = marl.build_model(env, algo, model_config) 
    result = algo.fit(env, model, stop=stop, **fit_kwargs) 
    return summarize_result(result)

def main(): 
    parser = argparse.ArgumentParser(description="Compare IPPO variants with configurable env, CUDA, and hyperparameters.") 
    parser.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS) 
    parser.add_argument("--hyperparam-source", default="test") 
    parser.add_argument("--environment-name", default="mpe") 
    parser.add_argument("--map-name", default="simple_spread") 
    parser.add_argument("--iterations", type=int, default=2) 
    parser.add_argument("--stop-timesteps", type=int, default=None) 
    parser.add_argument("--stop-reward", type=float, default=None) 
    parser.add_argument("--force-coop", dest="force_coop", action="store_true") 
    parser.add_argument("--no-force-coop", dest="force_coop", action="store_false") 
    parser.set_defaults(force_coop=True) 
    parser.add_argument("--core-arch", default="mlp") 
    parser.add_argument("--encode-layer", default="128-128") 
    parser.add_argument("--local-mode", dest="local_mode", action="store_true") 
    parser.add_argument("--no-local-mode", dest="local_mode", action="store_false") 
    parser.set_defaults(local_mode=True) 
    parser.add_argument("--num-gpus", type=float, default=0) 
    parser.add_argument("--num-workers", type=int, default=0) 
    parser.add_argument("--share-policy", default="group") 
    parser.add_argument("--checkpoint-freq", type=int, default=1) 
    parser.add_argument("--algo-arg", action="append", default=[]) 
    parser.add_argument("--env-arg", action="append", default=[]) 
    args = parser.parse_args() 
 
    stop = dict(DEFAULT_STOP) 
    stop["training_iteration"] = args.iterations 
    if args.stop_timesteps is not None: 
        stop["timesteps_total"] = args.stop_timesteps 
    if args.stop_reward is not None: 
        stop["episode_reward_mean"] = args.stop_reward 
 
    model_config = { 
        "core_arch": args.core_arch, 
        "encode_layer": args.encode_layer, 
    } 
    fit_kwargs = { 
        "local_mode": args.local_mode, 
        "num_gpus": args.num_gpus, 
        "num_workers": args.num_workers, 
        "share_policy": args.share_policy, 
        "checkpoint_freq": args.checkpoint_freq, 
    } 
    algo_overrides = parse_kv_list(args.algo_arg) 
    extra_env_args = parse_kv_list(args.env_arg)

    all_results = {} 
    for algo_name in args.algos: 
        print() 
        print(f"=== Running {algo_name} ===") 
        try: 
            summary = run_once(algo_name, args, stop, model_config, fit_kwargs, algo_overrides, extra_env_args) 
            all_results[algo_name] = summary 
        except Exception as exc: 
            all_results[algo_name] = {"error": str(exc)} 
        print(json.dumps(all_results[algo_name], indent=2)) 
 
    print() 
    print("=== Final Summary ===") 
    print(json.dumps(all_results, indent=2)) 
 
if __name__ == "__main__": 
    main() 
