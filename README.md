# Sample Efficiency in Sparse-Reward Navigation: Model-Free vs. Model-Based RL

**Pranav Manimaran, Vandana Mansur**
EEC 289A — Sensorimotor Learning, UC Davis, Spring 2026

## Overview

This project benchmarks three reinforcement learning agents — PPO (model-free), Neural Dyna-Q (classical model-based), and Dreamer (world-model-based) — on four MiniGrid sparse-reward navigation environments. We measure sample efficiency (steps to 50%/80% success), wall-clock training time, and zero-shot generalization to larger (16×16) grid variants.

## Environments

| Key | Environment ID | Challenge |
|-----|---------------|-----------|
| `empty` | `MiniGrid-Empty-8x8-v0` | Sanity check — all agents should solve this |
| `doorkey` | `MiniGrid-DoorKey-8x8-v0` | Two-stage: pick up key, unlock door, reach goal |
| `multiroom` | `MiniGrid-MultiRoom-N4-S5-v0` | Traverse 4 connected rooms — long credit assignment |
| `keycorridor` | `MiniGrid-KeyCorridor-S3-R1-v0` | Longest horizon, multiple locked doors |

All environments use sparse reward (+1 at goal only, 0 otherwise) and full observability.

## Agents

| Agent | Description | Training Steps |
|-------|-------------|----------------|
| **PPO** | Clipped surrogate objective, GAE(λ=0.95), shared CNN encoder | 500K |
| **Neural Dyna-Q** | DQN + neural transition model, planning ratio k=5 (real + imagined Q-updates) | 500K |
| **Dreamer** | RSSM world model (GRU h=512, latent z=32), latent actor-critic, imagination horizon H=15 | 200K |

Each agent is evaluated with 3 seeds (42, 123, 777) across all 4 environments = 36 main training runs.

## Installation

```bash
pip install minigrid==2.3.1 gymnasium==0.29.1 torch torchvision \
    numpy matplotlib seaborn pandas scipy tqdm
```

Python 3.10+ required.

## Quick Start — Smoke Test

Verifies all three agents train and log correctly. Runs ~15 minutes on a GPU (T4 or better).

```bash
python local_test.py
```

All three agents should print `[PASS]`. Output CSVs are written to `results_local_test/logs/` and cleaned up automatically.

## Single Training Run

```bash
python benchmark.py \
    --agent ppo \
    --env doorkey \
    --seed 42 \
    --total_steps 500000 \
    --save_dir results \
    --eval_generalization
```

**All flags:**

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--agent` | `ppo`, `dynaq`, `dreamer` | required | Agent to train |
| `--env` | `empty`, `doorkey`, `multiroom`, `keycorridor` | required | Environment |
| `--seed` | int | required | Random seed |
| `--total_steps` | int | required | Environment steps to train for |
| `--save_dir` | path | required | Directory for logs, checkpoints, tables |
| `--planning_k` | int | 5 | Dyna-Q only: planning steps per real step |
| `--imagination_h` | int | 15 | Dreamer only: imagination rollout horizon |
| `--eval_generalization` | flag | off | After training, zero-shot eval on 16×16 variant |

Logs are written to `{save_dir}/logs/`, checkpoints to `{save_dir}/checkpoints/`.

## Full Benchmark (36 runs + ablations + visualizations)

```bash
bash run_experiments.sh
```

Estimated time: **12–24 hours on a T4 GPU** (Colab or equivalent). The script runs:

1. **Phase 1** — 36 main training runs (4 envs × 3 seeds × 3 agents) with generalization eval
2. **Phase 2** — Dyna-Q planning ratio ablation: k ∈ {0, 1, 10, 20} on DoorKey, seed 42
3. **Phase 3** — Dreamer horizon ablation: H ∈ {5, 10, 20, 25} on MultiRoom, seed 42
4. **Phase 4** — All visualizations via `visualize.py`

## Visualizations

```bash
python visualize.py --results_dir results/
```

Figures are saved to `results/plots/`:

| File | Description |
|------|-------------|
| `learning_curves_all.png` | Success rate vs. steps, all envs (2×2 grid, mean ± std) |
| `learning_curves_{env}.png` | Per-environment learning curves |
| `sample_efficiency_50pct.png` | Steps to 50% success, grouped bar chart |
| `sample_efficiency_80pct.png` | Steps to 80% success, grouped bar chart |
| `wallclock.png` | Wall-clock minutes to 80% success |
| `generalization_heatmap.png` | Zero-shot success on 16×16 (requires `--eval_generalization`) |
| `ablation_planning_ratio.png` | Dyna-Q k sweep (requires ablation runs) |
| `ablation_horizon.png` | Dreamer H sweep (requires ablation runs) |
| `loss_curves_{ppo,dynaq,dreamer}.png` | Per-agent loss diagnostics |
| `per_seed_variance.png` | Individual seed runs vs. mean on Empty-8x8 |
| `grad_norm_diagnostics.png` | Gradient norm EMA over training |

Run with `--figures 1,2,3` to generate only specific figures.

## Results Directory Structure

```
results/
  logs/                          # Per-run training logs (CSV)
    {agent}_{env}_seed{N}.csv
  checkpoints/                   # Saved model weights
    {agent}_{env}_seed{N}_step{K}.pt
  plots/                         # All generated figures (PNG)
  tables/
    summary_table.csv            # Aggregated metrics (12 rows, one per agent×env)
    generalization_results.csv   # Zero-shot 16×16 success rates
  ablation_k/                    # Dyna-Q planning ratio sweep
    logs/  checkpoints/
  ablation_h/                    # Dreamer horizon sweep
    logs/  checkpoints/
```

## CSV Log Columns

| Column | PPO | Dyna-Q | Dreamer |
|--------|:---:|:------:|:-------:|
| `step` | ✓ | ✓ | ✓ |
| `success_rate` | ✓ | ✓ | ✓ |
| `mean_ep_length` | ✓ | ✓ | ✓ |
| `mean_return` | ✓ | ✓ | ✓ |
| `wall_time_sec` | ✓ | ✓ | ✓ |
| `grad_norm` | ✓ | ✓ | ✓ |
| `policy_loss` | ✓ | — | — |
| `value_loss` | ✓ | — | — |
| `entropy` | ✓ | — | — |
| `q_loss` | — | ✓ | — |
| `model_loss` | — | ✓ | — |
| `epsilon` | — | ✓ | — |
| `wm_loss` | — | — | ✓ |
| `actor_loss` | — | — | ✓ |
| `critic_loss` | — | — | ✓ |
| `recon_loss` | — | — | ✓ |
| `kl_loss` | — | — | ✓ |

## Colab Notebook

`benchmark_colab.ipynb` is a self-contained Colab notebook with all training and visualization code.

- **§8 config cell** contains `SMOKE_TEST = True/False` — the single place to switch modes
  - `True`: trains on Empty-8x8 only for 25K/25K/15K steps (~15 min on T4, verifies setup)
  - `False`: full 36-run benchmark (~3–6 hours on T4)
- Figures generated in the notebook match `visualize.py` output exactly

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Seeds | 42, 123, 777 |
| CNN encoder output dim | 256 |
| RSSM GRU hidden dim | 512 |
| RSSM latent z dim | 32 |
| PPO learning rate | 3e-4 |
| PPO clip epsilon | 0.2 |
| Dyna-Q Q-net learning rate | 1e-4 |
| Dyna-Q model learning rate | 1e-3 |
| Dyna-Q planning ratio k | 5 |
| Dreamer world model learning rate | 6e-4 |
| Dreamer actor/critic learning rate | 8e-5 |
| Dreamer imagination horizon H | 15 |
| Discount γ | 0.99 |
| GAE / lambda-return λ | 0.95 |
