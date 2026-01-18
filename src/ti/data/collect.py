import math
from typing import Tuple

import torch

from ti.data.buffer import ReplayBufferPlus
from ti.envs import TeacupMaze, make_env


@torch.no_grad()
def collect_offline_dataset(env_ctor, total_transitions, num_envs, maze_cfg, device) -> Tuple[ReplayBufferPlus, object]:
    iters = int(math.ceil(float(total_transitions) / float(num_envs)))
    buf = ReplayBufferPlus(
        obs_dim=maze_cfg["obs_dim"],
        max_size=iters * num_envs + 2048,
        num_envs=num_envs,
        device=device,
    )
    env = make_env(env_ctor, num_envs=num_envs, maze_cfg=maze_cfg, device=device)
    obs = env.reset()

    for _ in range(iters):
        a = torch.randint(0, maze_cfg["n_actions"], (num_envs,), device=device)

        if isinstance(env, TeacupMaze):
            nuis = env.current_nuis()
            special = env.current_special()
        else:
            nuis = getattr(env, "phase", getattr(env, "qptr", None))
            if nuis is None:
                nuis = torch.zeros((num_envs,), device=device, dtype=torch.long)
            special = torch.zeros((num_envs,), device=device, dtype=torch.bool)

        sp, done, reset_obs = env.step(a)
        buf.add_batch(obs, a, sp, done, nuis=nuis, special=special)
        obs = reset_obs
    return buf, env
