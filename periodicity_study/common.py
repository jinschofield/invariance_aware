import os
from typing import Tuple

import torch

from ti.envs import layouts


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def maze_cfg_from_config(cfg) -> dict:
    return {
        "maze_size": int(cfg.maze_size),
        "obs_dim": int(cfg.obs_dim),
        "n_actions": int(cfg.n_actions),
        "max_ep_steps": int(cfg.max_ep_steps),
        "goal": list(cfg.goal),
        "periodic_P": int(cfg.periodic_P),
        "slippery_D": int(getattr(cfg, "slippery_D", 3)),
        "teacup_P": int(getattr(cfg, "teacup_P", 4)),
    }


def free_positions(maze_size: int, device: torch.device) -> torch.Tensor:
    layout = layouts.make_layout(maze_size, device)
    return torch.nonzero(~layout).long()


def free_positions_for_env(env_id: str, maze_size: int, device: torch.device) -> torch.Tensor:
    if env_id.startswith("teacup"):
        layout = layouts.make_open_plate_layout(maze_size, device)
    else:
        layout = layouts.make_layout(maze_size, device)
    return torch.nonzero(~layout).long()


def build_obs_from_pos_phase(
    pos_rc: torch.Tensor, phase: torch.Tensor, maze_size: int, period: int
) -> torch.Tensor:
    xy = layouts.pos_norm_from_grid(pos_rc, maze_size)
    ph = layouts.phase_sincos3(phase, period)
    return torch.cat([xy, ph], dim=-1)


def build_obs_from_pos_nuisance(
    env_id: str, pos_rc: torch.Tensor, nuis: torch.Tensor, maze_cfg: dict, device: torch.device
) -> torch.Tensor:
    maze_size = int(maze_cfg["maze_size"])
    xy = layouts.pos_norm_from_grid(pos_rc, maze_size)

    if env_id.startswith("periodicity"):
        ph = layouts.phase_sincos3(nuis, int(maze_cfg["periodic_P"]))
        return torch.cat([xy, ph], dim=-1)

    if env_id.startswith("slippery"):
        ph = layouts.one_hot3(nuis.long())
        return torch.cat([xy, ph], dim=-1)

    if env_id.startswith("teacup"):
        centers = torch.tensor(
            [[3, 3], [3, 8], [8, 3], [8, 8], [6, 6]],
            device=device,
            dtype=torch.long,
        )
        rad2 = 2
        d2 = ((pos_rc[:, None, :] - centers[None, :, :]).float().pow(2)).sum(dim=-1)
        min_d2, cid = d2.min(dim=1)
        in_cup = min_d2 <= float(rad2)
        local_phase = (nuis.long() + cid) % int(maze_cfg["teacup_P"])
        feat = layouts.TETRA4.to(device)[local_phase] * in_cup.float().unsqueeze(1)
        return torch.cat([xy, feat], dim=-1)

    raise ValueError(f"Unknown env_id: {env_id}")
