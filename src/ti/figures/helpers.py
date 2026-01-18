def build_maze_cfg(cfg):
    maze = cfg.get("maze", {})
    return {
        "maze_size": int(maze.get("size", 12)),
        "obs_dim": int(maze.get("obs_dim", 5)),
        "n_actions": int(maze.get("n_actions", 4)),
        "max_ep_steps": int(maze.get("max_ep_steps", 60)),
        "goal": maze.get("goal", [10, 10]),
        "periodic_P": int(maze.get("periodic_P", 8)),
        "slippery_D": int(maze.get("slippery_D", 3)),
        "teacup_P": int(maze.get("teacup_P", 4)),
    }


def get_env_spec(cfg, env_id):
    envs = cfg.get("envs", {})
    if env_id not in envs:
        raise ValueError(f"Unknown env id: {env_id}")
    spec = dict(envs[env_id])
    return spec
