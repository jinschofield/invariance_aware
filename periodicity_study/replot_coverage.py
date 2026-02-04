import argparse
import csv
import glob
import os
from typing import Dict, Iterable, List, Tuple

from periodicity_study.config import StudyConfig
from periodicity_study.plotting import plot_multi_timeseries
from periodicity_study.run_study import _build_env_specs, _with_env_title


def _read_metrics(path: str) -> List[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_series(log_dir: str, policy: str) -> Tuple[Dict[str, Iterable[dict]], Dict[str, Iterable[dict]]]:
    intrinsic: Dict[str, Iterable[dict]] = {}
    goal: Dict[str, Iterable[dict]] = {}
    pattern = os.path.join(log_dir, f"metrics_timeseries_*_{policy}.csv")
    for path in glob.glob(pattern):
        name = os.path.basename(path).replace("metrics_timeseries_", "").replace(f"_{policy}.csv", "")
        rows = _read_metrics(path)
        if not rows:
            continue
        if name.endswith("_goal"):
            goal[name] = rows
        else:
            intrinsic[name] = rows
    return intrinsic, goal


def _order_series(series: Dict[str, Iterable[dict]], order: List[str]) -> Dict[str, Iterable[dict]]:
    if not series:
        return series
    ordered = {k: series[k] for k in order if k in series}
    for k, v in series.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def _plot_coverage(
    series_by_rep: Dict[str, Iterable[dict]],
    title: str,
    out_path: str,
) -> None:
    if not series_by_rep:
        return
    plot_multi_timeseries(
        series_by_rep,
        title=title,
        out_path=out_path,
        y_key="coverage_percent",
        y_label="Coverage (%)",
        x_key="steps_per_state",
        x_label="Steps per free state",
        hline_y=100.0,
        xscale="log",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-root",
        default="periodicity_study/outputs_sweep",
        help="Root containing comb_* folders.",
    )
    args = parser.parse_args()

    cfg = StudyConfig()
    env_name_by_id = {spec["id"]: spec["name"] for spec in _build_env_specs(cfg)}

    base_order = [
        "coord_only",
        "coord_plus_nuisance",
        "crtr_learned",
        "idm_learned",
        "crtr_online_joint",
        "idm_online_joint",
    ]
    goal_order = [f"{name}_goal" for name in base_order]

    comb_dirs = sorted(glob.glob(os.path.join(args.outputs_root, "comb_*")))
    for comb_dir in comb_dirs:
        log_root = os.path.join(comb_dir, "logs")
        fig_root = os.path.join(comb_dir, "figures")
        if not os.path.isdir(log_root):
            continue
        for env_id in sorted(os.listdir(log_root)):
            log_dir = os.path.join(log_root, env_id)
            if not os.path.isdir(log_dir):
                continue
            fig_dir = os.path.join(fig_root, env_id)
            os.makedirs(fig_dir, exist_ok=True)
            env_label = env_name_by_id.get(env_id, env_id)
            for policy in ("rep", "raw"):
                intrinsic, goal = _load_series(log_dir, policy)
                intrinsic = _order_series(intrinsic, base_order)
                goal = _order_series(goal, goal_order)

                if intrinsic:
                    title = _with_env_title(
                        f"PPO coverage over time (incl. online CRTR/IDM) ({policy})",
                        env_label,
                    )
                    out_path = os.path.join(
                        fig_dir, f"ppo_coverage_over_time_with_online_{policy}.png"
                    )
                    _plot_coverage(intrinsic, title, out_path)

                if goal:
                    title = _with_env_title(
                        f"PPO coverage over time (incl. online CRTR/IDM) + goal ({policy})",
                        env_label,
                    )
                    out_path = os.path.join(
                        fig_dir, f"ppo_coverage_over_time_with_online_goal_{policy}.png"
                    )
                    _plot_coverage(goal, title, out_path)


if __name__ == "__main__":
    main()
