# Periodicity Study (Standalone Subfolder)

This subfolder contains a standalone study that runs across multiple environments
(periodicity, delay action queue, teacup) and large-maze variants. It implements
three stages of measurement across three representation types:

1) Representation invariance (same position, different nuisance).
2) Elliptical bonus behavior (fixed position, across nuisance).
3) PPO action distributions (bonus-only reward).

Representations evaluated:
- Coord-only (normalized x,y).
- Coord + nuisance (x,y + phase index).
- CRTR learned representation (offline-trained).
- IDM learned representation (offline-trained).
- CRTR learned representation (online joint training with PPO + bonus).
- IDM learned representation (online joint training with PPO + bonus).

## Quick start

```bash
python -m periodicity_study.run_study
```

### Selecting environments
You can filter which environments run:

```bash
# Only large mazes
python -m periodicity_study.run_study --only-large

# Only base (small) mazes
python -m periodicity_study.run_study --only-small

# Specific env ids (comma-separated)
python -m periodicity_study.run_study --envs periodicity_large
python -m periodicity_study.run_study --envs periodicity,slippery
```

### Selecting reward mode (PPO only)
All PPO figures include all reps by default. You can still choose reward mode:

```bash
# Only goal (extrinsic) PPO
python -m periodicity_study.run_study --goal-only

# Only intrinsic PPO runs
python -m periodicity_study.run_study --intrinsic-only
```

### PPO policy inputs
By default PPO runs twice (representation embeddings + raw observations). To restrict:

```bash
python -m periodicity_study.run_study --policy-inputs raw
```

### Intrinsic mixing toggles
All intrinsic mixing optimizations are off by default. Enable as needed:

```bash
# Success-conditioned alpha annealing
python -m periodicity_study.run_study --use-alpha-anneal

# Episode-gated intrinsic after first extrinsic hit
python -m periodicity_study.run_study --use-alpha-gate

# Separate ext/int critics (disallowed for extrinsic runs by default)
python -m periodicity_study.run_study --use-two-critic

# Normalize (and optionally clip) intrinsic reward
python -m periodicity_study.run_study --use-int-norm --int-clip 5.0
```

This study is GPU-first and will error if CUDA is not available. Use `--device cuda:0`
to select a specific GPU.

## SLURM

```bash
sbatch periodicity_study/slurm/periodicity_study.sbatch
sbatch periodicity_study/slurm/periodicity_study_fast.sbatch
```

## Colab notebook

Notebook path: `periodicity_study/notebooks/periodicity_study_colab.ipynb`

Outputs (figures + CSVs) are written under:
`periodicity_study/outputs/`, organized by environment (subfolders per env).

## Online training logs

During PPO training, metrics for levels (1), (2), and (3) are logged over time:

- `periodicity_study/outputs/logs/metrics_timeseries_<rep>_<policy>.csv`
- `periodicity_study/outputs/logs/metrics_timeseries_crtr_online_joint_<policy>.csv`
