# Temporal Invariance (ICML figures)

This repo generates all figure assets for the ICML draft using the Periodicity, Slippery, and Teacup mazes (rotation maze removed).

## Quick start (local)
```
pip install -r requirements.txt
python scripts/run_figures.py --config configs/paper.yaml
```

## Colab
```
!pip install -r requirements.txt
!pip install -e .
```

Then open `notebooks/icml_figures.ipynb` and run the cells. The notebook only calls into the Python modules.

## Outputs
- Figures: `outputs/figures/<run_id>/`
- Tables: `outputs/tables/<run_id>/`
- Runs: `outputs/runs/<run_id>/` (checkpoints + logs + config snapshot)
- Cache: `outputs/cache/` (datasets and episode indices)

## Configs
- `configs/paper.yaml` selects envs, methods, and figure lineup
- `configs/envs.yaml` defines the three environments (Periodicity, Slippery, Teacup)
- `configs/methods.yaml` defines training/probe/elliptical settings
- `configs/figures.yaml` controls figure generation order

## Caching and checkpoints
- Datasets are cached under `outputs/cache/` when `runtime.cache_datasets=true`.
- Training losses are saved under `outputs/runs/<run_id>/logs/`.
- Checkpoints are saved under `outputs/runs/<run_id>/checkpoints/`.

## Throughput profiles
- `runtime.throughput_mode` controls when max‑throughput settings apply:
  - `rl_only`: only for figures tagged with `profile: rl`
  - `always`: always apply RL profile
  - `never`: never apply RL profile
- Use `runtime.rl_profile` to define the max‑throughput settings.

## Optuna (hyperparameter search)
```
python scripts/optuna_search.py --config configs/paper.yaml --optuna-config configs/optuna.yaml
```
