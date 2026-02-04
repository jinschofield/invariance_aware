# Sweep Context and Methodology

This document is a complete, self-contained description of the sweep that produced
`stability_success_report.csv` so an LLM (or new collaborator) can interpret every
configuration and metric without prior knowledge.

---

## 1) What this sweep is

We run PPO agents in several gridworld-style maze environments. Each run mixes:

- **Extrinsic reward**: success event (goal reached).
- **Intrinsic reward**: an *episodic elliptical bonus* computed in representation space.

We systematically toggle four optional stabilization techniques (A/B/C/D) and
evaluate **all representations**, **both policy input types**, and **all environments**.

The result is a comprehensive grid:

```
env × policy-input × (A/B/C/D combo) × representation
```

The CSV report aggregates these into a per‑combo summary (best rep, average success,
and stability statistics).

---

## 2) Environments

Each environment has a “small” and “large” version:

| ID | Description |
|---|---|
| `periodicity` | Base periodic maze |
| `slippery` | Delay action queue maze |
| `teacup` | Teacup maze |
| `periodicity_large` | Larger bottlenecked periodic maze |
| `slippery_large` | Larger bottlenecked delay‑action maze |
| `teacup_large` | Larger bottlenecked teacup maze |

**Large** versions are more difficult (more bottlenecks, longer paths), and typically
have lower success rates.

---

## 3) Policy input types

Each PPO policy can take one of two input types:

| Policy input | Meaning |
|---|---|
| `rep` | PPO policy receives the *representation embedding* \(z\) |
| `raw` | PPO policy receives the *raw observation* vector |

Both modes are run for every environment and every configuration. This isolates
whether the policy benefits from learned features vs raw state.

---

## 4) Representations (reps)

For every run, PPO is trained with **all** reps below. The CSV reports the best one.

| Rep name | Type | Description |
|---|---|---|
| `coord_only` | handcrafted | \((x,y)\) position only |
| `coord_plus_nuisance` | handcrafted | \((x,y)\) + nuisance variable (phase/queue/etc.) |
| `crtr_learned` | learned offline | CRTR rep learned from offline buffer |
| `idm_learned` | learned offline | IDM rep learned from offline buffer |
| `crtr_online_joint` | learned online | CRTR updated during PPO |
| `idm_online_joint` | learned online | IDM updated during PPO |

**Interpretation**:
- Handcrafted reps reveal how far simple features go.
- Learned reps test whether representation learning improves PPO.
- Online reps test whether joint training improves over fixed features.

---

## 5) Rewards

### Extrinsic reward
The agent receives a reward when its next state reaches the goal position.

### Intrinsic reward (elliptical bonus)
The bonus is computed from a Mahalanobis distance in feature space:

1. Build a feature vector:
   \[
   \phi(z, a) = [z; \text{onehot}(a)]
   \]
2. Maintain a per‑episode inverse covariance \(A^{-1}\) using a Sherman‑Morrison update.
3. Compute bonus:
   \[
   r^{int}_t = \beta \sqrt{\phi^\top A^{-1} \phi}
   \]

This encourages novel state‑action pairs.

---

## 6) A / B / C / D configurations (stability variants)

Each run toggles a subset of the following. All are **off by default**.

### **A — Success‑conditioned alpha annealing**
`--use-alpha-anneal`

Maintain an exponential moving average (EMA) of success rate \(\hat{p}_{succ}\), and
scale intrinsic weight:

\[
\alpha_t = \alpha_0 (1 - \hat{p}_{succ})^\eta
\]

**Purpose**: intrinsic dominates early, fades as agent succeeds.

---

### **B — Episode‑gated intrinsic**
`--use-alpha-gate`

Within each episode, once extrinsic success happens, set \(\alpha_t = 0\) for the rest
of the episode.

**Purpose**: avoids post‑goal novelty chasing.

---

### **C — Two‑critic decomposition**
`--use-two-critic`

Maintain separate value functions:

\[
V(s) \approx V_{ext}(s) + \alpha_t V_{int}(s)
\]

Compute advantages separately:

\[
A_t = A_t^{ext} + \alpha_t A_t^{int}
\]

**Important note**: for *extrinsic* runs in this study, the code forces **single‑critic**
to keep one unified value loss, even if C is enabled. (C is used for intrinsic‑only
cases or analysis, but not for extrinsic PPO in this sweep.)

---

### **D — Intrinsic normalization / clipping**
`--use-int-norm` (and optional `--int-clip`)

Normalize intrinsic reward by running std:

\[
\tilde{r}^{int}_t = \frac{r^{int}_t}{\hat{\sigma}(r^{int}) + \epsilon}
\]

Optionally clip:

\[
\tilde{r}^{int}_t \leftarrow \mathrm{clip}(\tilde{r}^{int}_t, -c, c)
\]

**Purpose**: stabilize scale of intrinsic reward across time and across reps.

---

## 7) Flag encoding used in the sweep

Each configuration is labeled by a bitmask:

- **A** = bit 0 (`--use-alpha-anneal`)
- **C** = bit 1 (`--use-two-critic`)
- **B** = bit 2 (`--use-alpha-gate`)
- **D** = bit 3 (`--use-int-norm`)

Examples:

- `comb_0` → no flags enabled  
- `comb_1` → A  
- `comb_2` → C  
- `comb_4` → B  
- `comb_8` → D  
- `comb_9` → A + D  
- `comb_15` → A + B + C + D

The CSV reports `comb` and a `flags` field (concatenated letters) so you can see which
stabilizers were enabled.

---

## 8) PPO configuration (fixed across runs)

This sweep uses standard clipped PPO with categorical policy:

- 2‑layer MLP, Tanh activations
- GAE with \(\gamma=0.99\), \(\lambda=0.95\)
- Clipping \(\epsilon=0.2\)
- Entropy coefficient **0.0** (no entropy bonus)
- Value loss coefficient **0.5**
- Gradient norm clip **0.5**

**Important**: the only differences across runs are the A/B/C/D toggles, reps, and
policy input type.

---

## 9) What each CSV column means

The report file `stability_success_report.csv` has one row per:

```
environment × policy_input × A/B/C/D combo
```

It aggregates across all reps.

### Success‑rate summary

- **best_rep**: rep with highest success rate in this configuration.
- **best_success**: success rate for that rep.
- **second_success**: runner‑up rep’s success rate.
- **gap**: best_success − second_success (margin of dominance).
- **mean_success_reps**: mean success across all reps.

### Stability (computed from best_rep’s time series)

- **series_n**: number of evaluation points.
- **series_mean**: average success across training.
- **series_std**: volatility across full training.
- **series_tail_mean**: mean over last 25% of evaluations.
- **series_tail_std**: volatility in last 25% (late‑training stability).
- **series_max_dd**: maximum drawdown (peak‑to‑trough drop).

---

## 10) How to interpret the stability metrics

- **Low series_std**: stable learning overall.
- **Low series_tail_std**: stable late‑training (less oscillation).
- **High series_max_dd**: regression after peak success (instability).
- **Large gap**: one rep strongly outperforms the rest in that config.

---

## 11) Sweep outputs on disk

Each sweep configuration writes to:

```
periodicity_study/outputs_sweep/comb_<IDX>/
```

Within each, outputs are separated by environment:

```
tables/<env>/
logs/<env>/
figures/<env>/
```

The stability report is at:

```
periodicity_study/outputs_sweep/summary/stability_success_report.csv
```

---

## 12) Practical next‑step guidance (for planning)

The CSV lets you decide between:

- **Max success** vs **stability** (use best_success vs tail_std).
- **Raw vs rep** sensitivity (compare policy input).
- **A/B/C/D effect** (compare rows by flags).
- **IDM vs CRTR** (see which rep becomes best_rep in each config).

A common workflow:

1. Filter rows to an environment of interest.
2. Identify best_success and small tail_std.
3. Compare flags to see which stabilizers improve performance or stability.
4. Compare best_rep to see if IDM or CRTR is favored.

