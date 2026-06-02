# Sample Efficiency in Sparse-Reward Navigation: Model-Free vs. Model-Based RL

**Pranav Manimaran, Vandana Mansur**  
EEC 289A — Sensorimotor Learning, UC Davis, Spring 2026

---

## Overview

This project benchmarks three reinforcement learning agents — **PPO** (model-free), **Neural Dyna-Q** (classical model-based), and **Dreamer** (world-model-based) — on four MiniGrid sparse-reward navigation environments. We measure sample efficiency (steps to 50%/80% success), wall-clock training time, and zero-shot generalization to larger (16×16) grid variants.

---

## Repository Structure

```
EEC-289A-Final-Project/
├── src/
│   ├── benchmark.py          # All three agents + training/eval/logging
│   ├── visualize.py          # Plot generation (all figures)
│   └── local_test.py         # Smoke test (~2 min on CPU)
│
├── notebooks/
│   ├── benchmark_colab.ipynb       # Google Colab (single GPU)
│   ├── benchmark_kaggle.ipynb      # Kaggle (CPU)
│   └── benchmark_kaggle_gpu.ipynb  # Kaggle (dual T4, parallel runs)
│
├── scripts/
│   └── run_experiments.sh    # Full 36-run + ablation launcher (local)
│
├── results/
│   ├── logs/                 # Per-run training CSVs
│   ├── plots/                # Generated figures (PNG)
│   └── tables/               # summary_table.csv, generalization_results.csv
│
├── reports/
│   ├── proposal.md                         # Project proposal
│   ├── plan.txt                            # Detailed experiment plan
│   ├── report_plan.md                      # Final report outline
│   ├── ppt_plan.md                         # Presentation outline
│   ├── run_notes.txt                       # Kaggle run notes & bug log
│   ├── initial_idea.txt                    # Initial brainstorm
│   └── SensorimotorLearning-Proposal.pdf   # Submitted proposal PDF
│
├── requirements.txt
└── README.md
```

---

## Environments

| Key | Environment ID | Challenge |
|-----|----------------|-----------|
| `empty` | `MiniGrid-Empty-8x8-v0` | Sanity check — navigate to goal |
| `doorkey` | `MiniGrid-DoorKey-8x8-v0` | Pick up key, unlock door, reach goal |
| `multiroom` | `MiniGrid-MultiRoom-N4-S5-v0` | Traverse 4 rooms — long credit assignment |
| `keycorridor` | `MiniGrid-KeyCorridorS3R1-v0` | Longest horizon, multiple locked doors |

All environments use **sparse reward** (+1 at goal only) and **full observability**.

Zero-shot generalization is evaluated on larger 16×16 variants after training.

---

## Agents

| Agent | Description | Steps |
|-------|-------------|-------|
| **PPO** | Clipped surrogate, GAE(λ=0.95), shared CNN encoder | 500K |
| **Neural Dyna-Q** | Double DQN + neural transition model, planning ratio k=5 | 500K |
| **Dreamer** | RSSM world model (GRU h=512, z=32), latent actor-critic, horizon H=15 | 200K |

**Experimental design:** 3 agents × 4 environments × 3 seeds (42, 123, 777) = **36 main runs**.

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ required. For GPU training, install the CUDA-enabled PyTorch wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Quick Start

### Smoke test (~2 min on CPU)

Runs all three agents for 1K steps on Empty-8x8, verifies CSV logging and checkpointing:

```bash
cd src
python local_test.py
```

All three agents should print `[PASS]`.

### Single training run

```bash
cd src
python benchmark.py \
    --agent ppo \
    --env doorkey \
    --seed 42 \
    --total_steps 500000 \
    --save_dir ../results \
    --eval_generalization
```

**CLI flags:**

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--agent` | `ppo`, `dynaq`, `dreamer` | required | Agent to train |
| `--env` | `empty`, `doorkey`, `multiroom`, `keycorridor` | required | Environment |
| `--seed` | int | required | Random seed |
| `--total_steps` | int | required | Environment steps |
| `--save_dir` | path | `results` | Output directory |
| `--planning_k` | int | 5 | Dyna-Q: imagined updates per real step |
| `--imagination_h` | int | 15 | Dreamer: imagination rollout horizon |
| `--eval_generalization` | flag | off | Zero-shot eval on 16×16 variant after training |

Logs → `{save_dir}/logs/`, checkpoints → `{save_dir}/checkpoints/`.

### Full benchmark (36 runs + ablations + figures)

```bash
bash scripts/run_experiments.sh
```

Estimated: **12–24 hours on a T4 GPU**. Runs all training phases, ablation sweeps, and generates all figures.

---

## Notebooks

All notebooks share the same codebase as `src/benchmark.py` and include a `SMOKE_TEST` flag in the config cell:

| Notebook | Platform | Parallelism |
|----------|----------|-------------|
| `benchmark_colab.ipynb` | Google Colab | Sequential, single GPU |
| `benchmark_kaggle.ipynb` | Kaggle | Sequential, CPU |
| `benchmark_kaggle_gpu.ipynb` | Kaggle | Parallel across dual T4 GPUs |

Set `SMOKE_TEST = True` to run a quick sanity check (~15 min on T4); `False` for the full benchmark.

---

## Visualization

```bash
cd src
python visualize.py --results_dir ../results
```

Figures are saved to `results/plots/`:

| File | Description |
|------|-------------|
| `fig1_learning_curves_2x2.png` | Success rate vs. steps, all envs (mean ± std) |
| `sample_efficiency_50pct.png` | Steps to 50% success, grouped bar chart |
| `sample_efficiency_80pct.png` | Steps to 80% success, grouped bar chart |
| `wallclock.png` | Wall-clock minutes to 80% success |
| `loss_curves_{agent}.png` | Per-agent loss diagnostics |
| `per_seed_variance.png` | Individual seed runs vs. mean on Empty-8×8 |
| `grad_norm_diagnostics.png` | Gradient norm EMA over training |
| `episode_length_curves.png` | Mean episode length over training |

---

## CSV Log Columns

| Column | PPO | Dyna-Q | Dreamer |
|--------|:---:|:------:|:-------:|
| `step` | ✓ | ✓ | ✓ |
| `success_rate` | ✓ | ✓ | ✓ |
| `mean_ep_length` | ✓ | ✓ | ✓ |
| `mean_return` | ✓ | ✓ | ✓ |
| `wall_time_sec` | ✓ | ✓ | ✓ |
| `grad_norm` | ✓ | ✓ | ✓ |
| `policy_loss` / `value_loss` / `entropy` | ✓ | — | — |
| `q_loss` / `model_loss` / `epsilon` | — | ✓ | — |
| `wm_loss` / `actor_loss` / `critic_loss` / `recon_loss` / `kl_loss` | — | — | ✓ |

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Seeds | 42, 123, 777 |
| CNN encoder output dim | 256 |
| RSSM GRU hidden dim (h) | 512 |
| RSSM latent dim (z) | 32 |
| PPO learning rate | 3×10⁻⁴ |
| PPO clip ε | 0.2 |
| Dyna-Q Q-net / model learning rate | 1×10⁻⁴ / 1×10⁻³ |
| Dyna-Q planning ratio k | 5 |
| Dreamer world-model / actor-critic lr | 6×10⁻⁴ / 8×10⁻⁵ |
| Dreamer imagination horizon H | 15 |
| Discount γ | 0.99 |
| GAE / lambda-return λ | 0.95 |

---

## Authors

- **Pranav Manimaran** — [pranavnm09123@gmail.com](mailto:pranavnm09123@gmail.com)
- **Vandana Mansur** — [vandanacmansur@gmail.com](mailto:vandanacmansur@gmail.com)
