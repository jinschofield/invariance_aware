import argparse
import os
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import matplotlib.pyplot as plt

from ti.online.intrinsic import EpisodicEllipticalBonus
from ti.metrics.elliptical import elliptical_bonus


def build_state_onehots(grid_size: int, device: torch.device) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    positions = []
    onehots = []
    for r in range(grid_size):
        for c in range(grid_size):
            idx = r * grid_size + c
            vec = torch.zeros((grid_size * grid_size,), device=device)
            vec[idx] = 1.0
            positions.append((r, c))
            onehots.append(vec)
    return torch.stack(onehots, dim=0), positions


def step_pos(pos: Tuple[int, int], action: int, grid_size: int) -> Tuple[int, int]:
    r, c = pos
    if action == 0:  # up
        r -= 1
    elif action == 1:  # down
        r += 1
    elif action == 2:  # left
        c -= 1
    else:  # right
        c += 1
    r = int(np.clip(r, 0, grid_size - 1))
    c = int(np.clip(c, 0, grid_size - 1))
    return r, c


def compute_heatmap(
    Ainv: torch.Tensor,
    state_onehots: torch.Tensor,
    n_actions: int,
    beta: float,
    avg_actions: bool,
    grid_size: int,
) -> torch.Tensor:
    if avg_actions:
        actions = torch.arange(n_actions, device=state_onehots.device)
        z_rep = state_onehots.repeat_interleave(n_actions, dim=0)
        a_rep = actions.repeat(state_onehots.shape[0])
        a_onehot = torch.nn.functional.one_hot(a_rep.long(), num_classes=n_actions).float()
        phi = torch.cat([z_rep, a_onehot], dim=1)
        b = elliptical_bonus(phi, Ainv, beta=float(beta))
        b = b.view(state_onehots.shape[0], n_actions).mean(dim=1)
    else:
        a_rep = torch.zeros((state_onehots.shape[0],), device=state_onehots.device, dtype=torch.long)
        a_onehot = torch.nn.functional.one_hot(a_rep, num_classes=n_actions).float()
        phi = torch.cat([state_onehots, a_onehot], dim=1)
        b = elliptical_bonus(phi, Ainv, beta=float(beta))
    return b.view(grid_size, grid_size)


def render_heatmap(
    heat: np.ndarray,
    pos: Tuple[int, int],
    title: str,
    vmin: float,
    vmax: float,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(3, 3))
    im = ax.imshow(heat, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.scatter([pos[1]], [pos[0]], c="red", s=80, marker="x")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_state_timeseries(
    heatmaps: List[np.ndarray],
    pos_trace: List[Tuple[int, int]],
    grid_size: int,
    out_path: str,
) -> None:
    t_steps = np.arange(len(heatmaps))
    heat_arr = np.stack(heatmaps, axis=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in range(grid_size):
        for c in range(grid_size):
            y = heat_arr[:, r, c]
            label = f"({r},{c})"
            ax.plot(t_steps, y, label=label)

    for t, (r, c) in enumerate(pos_trace):
        ax.scatter([t], [heat_arr[t, r, c]], c="black", s=30, marker="o", zorder=3)

    ax.set_title("Elliptical bonus per state over time")
    ax.set_xlabel("t")
    ax.set_ylabel("bonus")
    ax.grid(alpha=0.3)
    ax.legend(title="state", ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=2)
    parser.add_argument("--n-actions", type=int, default=4)
    parser.add_argument(
        "--no-avg-actions",
        action="store_true",
        help="Disable action-averaging when computing heatmaps.",
    )
    parser.add_argument(
        "--print-values",
        action="store_true",
        help="Print raw heatmap values at each timestep.",
    )
    parser.add_argument("--out-dir", type=str, default="elliptical_bonus_demo/outputs")
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    state_onehots, positions = build_state_onehots(args.grid_size, device)
    z_dim = state_onehots.shape[1]
    bonus = EpisodicEllipticalBonus(
        z_dim=z_dim,
        n_actions=args.n_actions,
        beta=args.beta,
        lam=args.lam,
        num_envs=1,
        device=device,
    )

    pos_idx = int(rng.integers(0, len(positions)))
    pos = positions[pos_idx]

    heatmaps = []
    pos_trace = []
    avg_actions = not args.no_avg_actions
    for t in range(int(args.steps)):
        Ainv = bonus.Ainv[0]
        heat = compute_heatmap(
            Ainv,
            state_onehots,
            args.n_actions,
            args.beta,
            avg_actions,
            args.grid_size,
        )
        heatmaps.append(heat.detach().cpu().numpy())
        pos_trace.append(pos)
        if args.print_values:
            heat_np = heat.detach().cpu().numpy()
            print(f"\n[t={t}] bonus heatmap:")
            print(np.array2string(heat_np, precision=6, suppress_small=False))

        action = int(rng.integers(0, args.n_actions))
        state_idx = pos[0] * args.grid_size + pos[1]
        z = state_onehots[state_idx].unsqueeze(0)
        a = torch.tensor([action], device=device)
        bonus.compute_and_update(z, a)
        pos = step_pos(pos, action, args.grid_size)

    vmin = float(np.min(heatmaps))
    vmax = float(np.max(heatmaps))

    frame_paths = []
    for t, (heat, cur_pos) in enumerate(zip(heatmaps, pos_trace)):
        out_path = os.path.join(frames_dir, f"heatmap_{t:04d}.png")
        render_heatmap(
            heat,
            cur_pos,
            title=f"Elliptical bonus (t={t})",
            vmin=vmin,
            vmax=vmax,
            out_path=out_path,
        )
        frame_paths.append(out_path)

    render_state_timeseries(
        heatmaps,
        pos_trace,
        args.grid_size,
        out_path=os.path.join(args.out_dir, "state_bonus_timeseries.png"),
    )

    images = [imageio.imread(p) for p in frame_paths]
    gif_path = os.path.join(args.out_dir, "bonus_evolution.gif")
    imageio.mimsave(gif_path, images, duration=1.0 / max(1, args.fps))
    print(f"Saved {len(frame_paths)} frames to {frames_dir}")
    print(f"Saved GIF to {gif_path}")


if __name__ == "__main__":
    main()
