from ti.envs.periodic import PeriodicMaze
from ti.envs.slippery import SlipperyDelayMaze
from ti.envs.teacup import TeacupMaze


ENV_REGISTRY = {
    "PeriodicMaze": PeriodicMaze,
    "SlipperyDelayMaze": SlipperyDelayMaze,
    "TeacupMaze": TeacupMaze,
}


def make_env(env_ctor, num_envs, maze_cfg, device):
    if isinstance(env_ctor, str):
        if env_ctor not in ENV_REGISTRY:
            raise ValueError(f"Unknown env ctor: {env_ctor}")
        env_cls = ENV_REGISTRY[env_ctor]
    else:
        env_cls = env_ctor

    return env_cls(
        num_envs=num_envs,
        maze_size=maze_cfg["maze_size"],
        max_ep_steps=maze_cfg["max_ep_steps"],
        n_actions=maze_cfg["n_actions"],
        goal=maze_cfg["goal"],
        device=device,
    )
