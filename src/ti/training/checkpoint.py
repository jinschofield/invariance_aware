import os
from typing import Any, Dict, Optional

import torch

from ti.utils import ensure_dir


def find_latest_checkpoint(ckpt_dir: str, prefix: str = "ckpt_step", suffix: str = ".pt"):
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return None, 0
    best_step = 0
    best_path = None
    for name in os.listdir(ckpt_dir):
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        stem = name[len(prefix) : -len(suffix)]
        try:
            step = int(stem)
        except ValueError:
            continue
        if step > best_step:
            best_step = step
            best_path = os.path.join(ckpt_dir, name)
    return best_path, best_step


def load_checkpoint(
    path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[torch.device] = None,
):
    payload = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    ensure_dir(os.path.dirname(path))
    payload = {
        "model_state_dict": model.state_dict(),
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
