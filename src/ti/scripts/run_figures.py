import argparse
import os
import time

import torch

from ti.config.defaults import load_config
from ti.figures.registry import get_generator
from ti.plotting.style import set_paper_style
from ti.utils import configure_torch, ensure_dir, seed_everything


import threading
import subprocess

def start_background_zipper(source_dir, zip_path, interval_minutes=15):
    """Periodically zips the output directory in a background thread."""
    def _zip_loop():
        while True:
            time.sleep(interval_minutes * 60)
            try:
                                                         
                print(f"[Backup] Zipping {source_dir} -> {zip_path} ...", flush=True)
                                                                                  
                subprocess.run(
                    ["zip", "-r", "-q", "-FS", zip_path, "."],
                    cwd=source_dir,
                    check=False
                )
                print(f"[Backup] Saved to {zip_path}", flush=True)
            except Exception as e:
                print(f"[Backup] Failed: {e}", flush=True)

    t = threading.Thread(target=_zip_loop, daemon=True)
    t.start()
    return t

def run_from_config(config_path, figure_id=None, seed_override=None):
    cfg = load_config(config_path)
    runtime = cfg["runtime"]
    if seed_override is not None:
        runtime["seed"] = int(seed_override)

    output_dir = runtime.get("output_dir", "outputs")
    cache_dir = runtime.get("cache_dir", os.path.join(output_dir, "cache"))
    run_id = runtime.get("run_id") or time.strftime("%Y%m%d_%H%M%S")

    runtime["output_dir"] = output_dir
    runtime["cache_dir"] = cache_dir
    runtime["run_id"] = run_id
    runtime["run_dir"] = os.path.join(output_dir, "runs", run_id)
    runtime["fig_dir"] = os.path.join(output_dir, "figures", run_id)
    runtime["table_dir"] = os.path.join(output_dir, "tables", run_id)
    runtime["log_dir"] = os.path.join(runtime["run_dir"], "logs")
    runtime["ckpt_dir"] = os.path.join(runtime["run_dir"], "checkpoints")

    ensure_dir(runtime["output_dir"])
    ensure_dir(runtime["cache_dir"])
    ensure_dir(runtime["run_dir"])
    ensure_dir(runtime["fig_dir"])
    ensure_dir(runtime["table_dir"])
    ensure_dir(runtime["log_dir"])
    ensure_dir(runtime["ckpt_dir"])

                       
    if runtime.get("auto_zip", True):
                                                                                    
                                                    
                                                                                        
        project_root = os.path.dirname(os.path.abspath(output_dir))
        zip_path = os.path.join(project_root, "results_latest.zip")
        interval = runtime.get("auto_zip_interval", 15)
        start_background_zipper(output_dir, zip_path, interval_minutes=interval)

    seed_everything(runtime.get("seed", 0), deterministic=runtime.get("deterministic", True))
    device = torch.device(runtime.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_torch(runtime, device)
    set_paper_style()

    figures = cfg.get("figures", {})
    runtime_base = dict(runtime)
    if runtime.get("save_config_snapshot", True):
        snapshot_path = os.path.join(runtime["run_dir"], "config_snapshot.yaml")
        with open(snapshot_path, "w", encoding="utf-8") as f:
            import yaml

            yaml.safe_dump(cfg, f, sort_keys=False)
    for fid, spec in figures.items():
        if figure_id is not None and fid != figure_id:
            continue
        profile = spec.get("profile", "offline")
        runtime.clear()
        runtime.update(runtime_base)
        mode = runtime_base.get("throughput_mode", "never")
        if (mode == "always") or (mode == "rl_only" and profile == "rl"):
            runtime.update(runtime_base.get("rl_profile", {}))
        configure_torch(runtime, device)
        gen_name = spec.get("generator")
        if not gen_name:
            raise ValueError(f"Figure {fid} missing generator field.")
        generator = get_generator(gen_name)
        try:
            generator(cfg, fid, spec)
        except Exception:
            if runtime.get("continue_on_error", False):
                print(f"[WARN] Figure {fid} failed; continuing.", flush=True)
                continue
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--fig", default=None, help="Figure id to run")
    parser.add_argument("--seed", default=None, type=int, help="Override RNG seed")
    args = parser.parse_args()
    run_from_config(args.config, figure_id=args.fig, seed_override=args.seed)


if __name__ == "__main__":
    main()
