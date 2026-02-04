import argparse
import os
from typing import List, Tuple

import numpy as np
import torch

from ti.data.buffer import build_episode_index_strided
from ti.data.collect import collect_offline_dataset
from ti.envs import PeriodicMaze, SlipperyDelayMaze, TeacupMaze
from ti.online.buffer import OnlineReplayBuffer
from ti.utils import configure_torch, seed_everything

from periodicity_study.common import ensure_dir, free_positions_for_env, maze_cfg_from_config
from periodicity_study.config import StudyConfig
from periodicity_study.metrics import build_bonus_heatmaps
from periodicity_study.plotting import plot_heatmap_fixed
from periodicity_study.ppo import train_ppo
from periodicity_study.representations import (
    CoordNuisanceRep,
    CoordOnlyRep,
    train_or_load_crtr,
    train_or_load_idm,
)
from periodicity_study.run_study import _apply_env_spec, _build_env_specs, _with_env_title, apply_fast_cfg


def _combo_flags(idx: int) -> Tuple[bool, bool, bool, bool]:
    use_alpha_anneal = bool(idx & 1)
    use_two_critic = bool(idx & 2)
    use_alpha_gate = bool(idx & 4)
    use_int_norm = bool(idx & 8)
    return use_alpha_anneal, use_two_critic, use_alpha_gate, use_int_norm


def _build_env_ctor(env_id: str):
    if env_id.startswith("slippery"):
        return SlipperyDelayMaze
    if env_id.startswith("teacup"):
        return TeacupMaze
    return PeriodicMaze


def _select_rep(
    name: str,
    env_id: str,
    nuis_count: int,
    cfg: StudyConfig,
    device: torch.device,
    model_dir: str,
    buf,
    epi,
):
    if name == "coord_only":
        return CoordOnlyRep()
    if name == "coord_plus_nuisance":
        return CoordNuisanceRep(env_id, nuis_count, device)
    if name == "crtr_learned":
        return train_or_load_crtr(cfg, device, model_dir, force_retrain=False, buf=buf, epi=epi)
    if name == "idm_learned":
        return train_or_load_idm(cfg, device, model_dir, force_retrain=False, buf=buf, epi=epi)
    raise ValueError(f"Unknown rep: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo", type=int, default=1, help="Sweep combo index (0-15).")
    parser.add_argument("--env", default="periodicity", help="Environment id.")
    parser.add_argument(
        "--rep",
        default="crtr_learned",
        choices=["coord_only", "coord_plus_nuisance", "crtr_learned", "idm_learned"],
        help="Representation to track.",
    )
    parser.add_argument("--policy", default="rep", choices=["rep", "raw"], help="Policy input.")
    parser.add_argument("--output-dir", default=None, help="Override output root.")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda:0).")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution even if require_cuda is True.",
    )
    parser.add_argument("--fast", action="store_true", help="Use smaller settings.")
    parser.add_argument("--every", type=int, default=2, help="Eval cadence in updates.")
    parser.add_argument("--max-frames", type=int, default=50, help="Max heatmap frames.")
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=1e-4,
        help="Skip frames with max abs diff below this threshold.",
    )
    args = parser.parse_args()

    cfg = StudyConfig()
    if args.fast:
        apply_fast_cfg(cfg)

    use_alpha_anneal, use_two_critic, use_alpha_gate, use_int_norm = _combo_flags(int(args.combo))
    cfg.ppo_use_alpha_anneal = use_alpha_anneal
    cfg.ppo_use_two_critic = use_two_critic
    cfg.ppo_use_alpha_gate = use_alpha_gate
    cfg.ppo_use_int_norm = use_int_norm

    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(base_dir)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if not os.path.isabs(cfg.output_dir):
        cfg.output_dir = os.path.join(repo_root, cfg.output_dir)

    if args.device:
        cfg.device = args.device
    if args.allow_cpu:
        cfg.require_cuda = False
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if cfg.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required. Pass --device cuda:0 or --allow-cpu.")
    configure_torch(
        {"deterministic": True, "allow_tf32": True, "matmul_precision": "high"},
        device,
    )
    seed_everything(int(cfg.seed), deterministic=True)

    env_specs = _build_env_specs(cfg)
    env_spec = next((s for s in env_specs if s["id"] == args.env), None)
    if env_spec is None:
        known = ", ".join(sorted(spec["id"] for spec in env_specs))
        raise ValueError(f"Unknown env id {args.env}. Known: {known}")
    _apply_env_spec(cfg, env_spec)

    env_id = env_spec["id"]
    env_label = f"{env_spec['name']} ({env_id})"
    env_ctor = _build_env_ctor(env_id)

    fig_dir = os.path.join(cfg.output_dir, "figures", env_id)
    model_dir = os.path.join(cfg.output_dir, "models", env_id)
    log_dir = os.path.join(cfg.output_dir, "logs", env_id)
    ensure_dir(fig_dir)
    ensure_dir(model_dir)
    ensure_dir(log_dir)

    maze_cfg = maze_cfg_from_config(cfg)
    free_count = int(free_positions_for_env(env_id, maze_cfg["maze_size"], device).shape[0])
    buf, _ = collect_offline_dataset(
        env_ctor,
        cfg.offline_collect_steps,
        cfg.offline_num_envs,
        maze_cfg,
        device,
    )
    epi = build_episode_index_strided(buf.timestep, buf.size, cfg.offline_num_envs, device)

    if env_id.startswith("slippery"):
        nuis_count = cfg.slippery_D
    elif env_id.startswith("teacup"):
        nuis_count = cfg.teacup_P
    else:
        nuis_count = cfg.periodic_P

    rep = _select_rep(args.rep, env_id, nuis_count, cfg, device, model_dir, buf, epi)

    policy_obs_fn = None
    policy_input_dim = None
    if args.policy == "raw":
        policy_obs_fn = lambda obs, rep_obs: obs
        policy_input_dim = int(cfg.obs_dim)

    eval_buf_size = int(cfg.online_eval_buffer_size)
    if eval_buf_size <= 0:
        eval_buf_size = int(cfg.ppo_total_steps) * int(cfg.ppo_num_envs)
    eval_buf = OnlineReplayBuffer(cfg.obs_dim, eval_buf_size, cfg.ppo_num_envs, device)

    heatmaps: List[torch.Tensor] = []
    heat_steps: List[int] = []
    heat_updates: List[int] = []

    def eval_cb(update, env_steps, model):
        if eval_buf.size >= int(cfg.online_eval_min_buffer):
            h_mean, _ = build_bonus_heatmaps(rep, eval_buf, cfg, device, env_id)
            if heatmaps:
                diff = torch.nan_to_num(h_mean - heatmaps[-1], nan=0.0).abs().max().item()
                if diff < float(args.diff_threshold):
                    return {}
            if len(heatmaps) < int(args.max_frames):
                heatmaps.append(h_mean.detach().cpu())
                heat_steps.append(int(env_steps))
                heat_updates.append(int(update))
        return {}

    train_ppo(
        rep,
        cfg,
        device,
        env_ctor,
        maze_cfg,
        policy_obs_fn=policy_obs_fn,
        policy_input_dim=policy_input_dim,
        use_extrinsic=False,
        eval_callback=eval_cb,
        eval_every_updates=int(args.every),
        eval_buffer=eval_buf,
    )

    if not heatmaps:
        raise RuntimeError("No heatmaps captured; try lowering --diff-threshold or --every.")

    heat_np = torch.stack(heatmaps, dim=0).numpy()
    vmin = float(np.nanmin(heat_np))
    vmax = float(np.nanmax(heat_np))

    frames_dir = os.path.join(
        fig_dir, f"bonus_evolution_frames_{args.rep}_{args.policy}_comb_{args.combo}"
    )
    ensure_dir(frames_dir)

    min_vals = []
    max_vals = []
    for idx, heat in enumerate(heatmaps):
        frame_path = os.path.join(frames_dir, f"bonus_{idx:04d}.png")
        plot_heatmap_fixed(
            heat,
            title=_with_env_title(
                f"Elliptical bonus (t={idx}) - {args.rep} ({args.policy})",
                env_label,
            ),
            out_path=frame_path,
            vmin=vmin,
            vmax=vmax,
        )
        arr = heat.numpy()
        mask = np.isfinite(arr)
        min_vals.append(float(np.min(arr[mask])) if mask.any() else float("nan"))
        max_vals.append(float(np.max(arr[mask])) if mask.any() else float("nan"))

    scale_csv = os.path.join(
        fig_dir, f"bonus_scale_over_time_{args.rep}_{args.policy}_comb_{args.combo}.csv"
    )
    with open(scale_csv, "w", newline="", encoding="utf-8") as f:
        f.write("update,env_steps,bonus_min,bonus_max\n")
        for update, steps, vmin_i, vmax_i in zip(heat_updates, heat_steps, min_vals, max_vals):
            f.write(f"{update},{steps},{vmin_i},{vmax_i}\n")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(heat_steps, min_vals, marker="o", label="bonus_min")
    ax.plot(heat_steps, max_vals, marker="o", label="bonus_max")
    ax.set_title(_with_env_title("Elliptical bonus scale over time", env_label))
    ax.set_xlabel("env_steps")
    ax.set_ylabel("bonus value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            fig_dir, f"bonus_scale_over_time_{args.rep}_{args.policy}_comb_{args.combo}.png"
        )
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
