from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RuntimeConfig:
    seed: int = 0
    device: Optional[str] = None
    output_dir: str = "outputs"
    cache_dir: str = "outputs/cache"
    deterministic: bool = True


@dataclass
class FigureSpec:
    figure_id: str
    generator: str
    title: Optional[str] = None
    envs: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PaperConfig:
    runtime: RuntimeConfig
    maze: Dict[str, Any]
    envs: Dict[str, Any]
    methods: Dict[str, Any]
    figures: Dict[str, FigureSpec]
