from ti.figures import fig_elliptical_heatmaps, fig_rep_sweep, fig_teacup_only


REGISTRY = {
    "rep_sweep": fig_rep_sweep.run,
    "elliptical_heatmaps": fig_elliptical_heatmaps.run,
    "teacup_elliptical_only": fig_teacup_only.run,
}


def get_generator(name):
    if name not in REGISTRY:
        raise ValueError(f"Unknown figure generator: {name}")
    return REGISTRY[name]
