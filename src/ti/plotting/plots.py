import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator


def _tight_ylim(ax, vals, pad_frac=0.15):
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if np.isclose(vmin, vmax):
        pad = 1e-3 if vmax == 0 else abs(vmax) * 0.05
    else:
        pad = (vmax - vmin) * pad_frac
    ax.set_ylim(vmin - pad, vmax + pad)


def _apply_axes_style(ax, max_ticks=4, grid_y=False):
    ax.tick_params(axis="both", which="major", length=3, width=0.8, pad=2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks))
    ax.grid(False)
    if grid_y:
        ax.yaxis.grid(True, alpha=0.2, linewidth=0.5)
    sns.despine(ax=ax, top=True, right=True)


def plot_env_bars(
    env_name,
    rows,
    metric_key,
    title,
    chance_line=None,
    save_path=None,
    higher_better=False,
    highlight_best=True,
):
    names = [r["method"] for r in rows]
    vals = np.array([r[metric_key] for r in rows], dtype=np.float64)
    plt.figure(figsize=(22, 4))
    base = sns.color_palette("colorblind", n_colors=max(6, len(names)))
    palette = [sns.desaturate(base[i], 0.6) for i in range(len(names))]
    if highlight_best and len(vals) > 0:
        best_idx = int(np.argmax(vals)) if higher_better else int(np.argmin(vals))
        palette[best_idx] = base[best_idx]
    sns.barplot(x=names, y=vals, palette=palette)
    if chance_line is not None:
        plt.axhline(chance_line, ls="--", lw=0.8, color="#888888")
    plt.title(f"{env_name}: {title}")
    ax = plt.gca()
    _tight_ylim(ax, vals)
    _apply_axes_style(ax, max_ticks=4, grid_y=False)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        save_paths = [save_path] if isinstance(save_path, str) else list(save_path)
        for path in save_paths:
            plt.savefig(path)
        plt.close()
    else:
        plt.show()


def plot_heatmap(heat, title, mask=None, save_path=None):
    h = heat.detach().cpu().numpy()
    if mask is not None:
        m = mask.detach().cpu().numpy().astype(bool)
        h[~m] = np.nan
    plt.figure(figsize=(6, 6))
    sns.heatmap(h, square=True, cmap="viridis", cbar=True)
    plt.title(title)
    ax = plt.gca()
    ax.tick_params(axis="both", which="major", length=2, width=0.6, pad=2)
    sns.despine(ax=ax, top=True, right=True)
    if save_path:
        save_paths = [save_path] if isinstance(save_path, str) else list(save_path)
        for path in save_paths:
            plt.savefig(path)
        plt.close()
    else:
        plt.show()


def plot_loss_curve(steps, losses, title, save_path=None):
    plt.figure(figsize=(6, 3))
    plt.plot(steps, losses)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("loss")
    ax = plt.gca()
    _apply_axes_style(ax, max_ticks=4, grid_y=False)
    plt.tight_layout()
    if save_path:
        save_paths = [save_path] if isinstance(save_path, str) else list(save_path)
        for path in save_paths:
            plt.savefig(path)
        plt.close()
    else:
        plt.show()
