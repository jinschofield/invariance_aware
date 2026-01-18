import matplotlib as mpl
import seaborn as sns


def set_paper_style():
    sns.set_theme(context="paper", style="white")
    sns.set_palette("colorblind")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def method_palette(methods, highlight_method=None):
    base = sns.color_palette("colorblind", n_colors=max(6, len(methods)))
    colors = [sns.desaturate(base[i], 0.6) for i in range(len(methods))]
    if highlight_method in methods:
        idx = methods.index(highlight_method)
        colors[idx] = base[idx]
    return {m: colors[i] for i, m in enumerate(methods)}
