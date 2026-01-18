import os
from typing import Optional, Tuple

import torch

from ti.data.buffer import EpisodeIndex, ReplayBufferPlus, build_episode_index_strided
from ti.utils import ensure_dir


def _buffer_to_dict(buf: ReplayBufferPlus) -> dict:
    return {
        "obs_dim": buf.obs_dim,
        "max_size": buf.max_size,
        "num_envs": buf.num_envs,
        "size": buf.size,
        "s": buf.s,
        "sp": buf.sp,
        "a": buf.a,
        "d": buf.d,
        "timestep": buf.timestep,
        "current_timestep": buf.current_timestep,
        "nuis": buf.nuis,
        "special": buf.special,
    }


def _buffer_from_dict(data: dict, device: torch.device) -> ReplayBufferPlus:
    buf = ReplayBufferPlus(
        obs_dim=data["obs_dim"],
        max_size=data["max_size"],
        num_envs=data["num_envs"],
        device=device,
    )
    buf.size = int(data["size"])
    buf.s[:] = data["s"].to(device)
    buf.sp[:] = data["sp"].to(device)
    buf.a[:] = data["a"].to(device)
    buf.d[:] = data["d"].to(device)
    buf.timestep[:] = data["timestep"].to(device)
    buf.current_timestep[:] = data["current_timestep"].to(device)
    buf.nuis[:] = data["nuis"].to(device)
    buf.special[:] = data["special"].to(device)
    return buf


def save_buffer(path: str, buf: ReplayBufferPlus, epi: Optional[EpisodeIndex] = None) -> None:
    ensure_dir(os.path.dirname(path))
    payload = {"buffer": _buffer_to_dict(buf)}
    if epi is not None:
        payload["episode_index"] = {
            "starts": epi.starts,
            "lengths": epi.lengths,
            "num_envs": epi.num_envs,
        }
    torch.save(payload, path)


def load_buffer(path: str, device: torch.device) -> Tuple[ReplayBufferPlus, Optional[EpisodeIndex]]:
    payload = torch.load(path, map_location=device)
    buf = _buffer_from_dict(payload["buffer"], device=device)
    epi = None
    epi_payload = payload.get("episode_index")
    if epi_payload is not None:
        epi = EpisodeIndex(
            epi_payload["starts"].to(device),
            epi_payload["lengths"].to(device),
            epi_payload["num_envs"],
        )
    return buf, epi


def load_or_build_episode_index(buf: ReplayBufferPlus, epi: Optional[EpisodeIndex], device: torch.device):
    if epi is not None:
        return epi
    return build_episode_index_strided(buf.timestep, buf.size, buf.num_envs, device)
