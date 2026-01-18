import argparse
import os

import torch

from ti.config.defaults import load_config
from ti.data.buffer import build_episode_index_strided
from ti.data.cache import save_buffer
from ti.data.collect import collect_offline_dataset
from ti.figures.helpers import build_maze_cfg, get_env_spec
from ti.utils import ensure_dir, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--seed", default=None, type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    runtime = cfg["runtime"]
    if args.seed is not None:
        runtime["seed"] = int(args.seed)

    seed_everything(runtime.get("seed", 0), deterministic=runtime.get("deterministic", True))
    device = torch.device(runtime.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    maze_cfg = build_maze_cfg(cfg)

    cache_dir = runtime.get("cache_dir", "outputs/cache")
    ensure_dir(cache_dir)

    for env_id in cfg.get("envs", {}):
        env_spec = get_env_spec(cfg, env_id)
        print(f"Caching env={env_id} ({env_spec['name']})")
        buf, _env = collect_offline_dataset(
            env_spec["ctor"],
            cfg["methods"]["train"]["offline_collect_steps"],
            cfg["methods"]["train"]["offline_num_envs"],
            maze_cfg,
            device,
        )
        epi = build_episode_index_strided(buf.timestep, buf.size, cfg["methods"]["train"]["offline_num_envs"], device)
        path = os.path.join(cache_dir, f"{env_id}_seed{runtime['seed']}.pt")
        save_buffer(path, buf, epi)


if __name__ == "__main__":
    main()
