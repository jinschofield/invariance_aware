import os
from typing import Any, Dict, Optional

import yaml


def _resolve_path(base_dir: str, path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(paper_config_path: str) -> Dict[str, Any]:
    base_dir = os.path.dirname(os.path.abspath(paper_config_path))
    paper = load_yaml(paper_config_path)

    includes = paper.get("includes", {})
    envs_path = _resolve_path(base_dir, includes.get("envs", "configs/envs.yaml"))
    methods_path = _resolve_path(base_dir, includes.get("methods", "configs/methods.yaml"))
    figures_path = _resolve_path(base_dir, includes.get("figures", "configs/figures.yaml"))

    envs_cfg = load_yaml(envs_path) if envs_path else {}
    methods_cfg = load_yaml(methods_path) if methods_path else {}
    figures_cfg = load_yaml(figures_path) if figures_path else {}

    return {
        "runtime": paper.get("runtime", {}),
        "envs": envs_cfg.get("envs", envs_cfg),
        "maze": envs_cfg.get("maze", {}),
        "methods": methods_cfg.get("methods", methods_cfg),
        "figures": figures_cfg.get("figures", figures_cfg),
    }
