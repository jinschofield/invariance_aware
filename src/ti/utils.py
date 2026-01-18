import os
import random
from typing import Optional

import numpy as np
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device(device_str: Optional[str] = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_torch(runtime: dict, device: torch.device) -> None:
    deterministic = bool(runtime.get("deterministic", True))
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    if device.type == "cuda":
        allow_tf32 = bool(runtime.get("allow_tf32", True))
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "tf32" if allow_tf32 else "ieee"
        else:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32

        if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
            torch.backends.cudnn.conv.fp32_precision = "tf32" if allow_tf32 else "ieee"
        else:
            torch.backends.cudnn.allow_tf32 = allow_tf32

        if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = bool(
                runtime.get("allow_fp16_reduced_precision_reduction", True)
            )
        if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bool(
                runtime.get("allow_bf16_reduced_precision_reduction", True)
            )

    precision = runtime.get("matmul_precision", None)
    if precision and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(precision)


def maybe_compile(model, runtime: dict):
    if runtime.get("compile", False) and hasattr(torch, "compile"):
        mode = runtime.get("compile_mode", "max-autotune")
        return torch.compile(model, mode=mode)
    return model


def get_amp_settings(runtime: dict, device: torch.device):
    use_amp = bool(runtime.get("use_amp", False)) and device.type == "cuda"
    amp_dtype = str(runtime.get("amp_dtype", "bf16")).lower()
    scaler = None
    if use_amp and amp_dtype in ("fp16", "float16"):
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    return use_amp, amp_dtype, scaler


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
