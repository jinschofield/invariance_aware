## Elliptical Bonus Demo (2x2 Grid)

This standalone demo simulates the **elliptical bonus** on a 2x2 wall-less grid
with **one-hot state representations**. It produces:

- A heatmap for **every time step** (saved as PNGs)
- A GIF showing the evolution of the heatmaps

### Run

```bash
python -m elliptical_bonus_demo.run_demo --steps 50 --out-dir elliptical_bonus_demo/outputs
```

### Output

```
<out-dir>/
  frames/
    heatmap_0000.png
    heatmap_0001.png
    ...
  bonus_evolution.gif
```
