#!/bin/bash
# run_experiments.sh — Launch all 36 training runs, ablation sweeps, and visualization.
# Estimated GPU time: ~12–24h on a T4 GPU (Colab) for full 36 runs.
# Usage: bash run_experiments.sh
set -e

PYTHON=python3
ENVS=("empty" "doorkey" "multiroom" "keycorridor")
SEEDS=(42 123 777)

mkdir -p results/logs results/checkpoints results/plots results/tables
mkdir -p results/ablation_k/logs results/ablation_k/checkpoints
mkdir -p results/ablation_h/logs results/ablation_h/checkpoints

# ==============================================================================
# Phase 1 — 36 main training runs (plan.txt §9)
# PPO / Dyna-Q: 500K steps each
# Dreamer:      200K steps each
# --eval_generalization: zero-shot eval on 16x16 variant after each run
# ==============================================================================
echo "========================================"
echo "  Phase 1: 36 main training runs"
echo "========================================"

for env in "${ENVS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "--- PPO | $env | seed $seed ---"
    $PYTHON benchmark.py \
      --agent ppo --env $env --seed $seed \
      --total_steps 500000 \
      --save_dir results \
      --eval_generalization

    echo "--- Dyna-Q | $env | seed $seed ---"
    $PYTHON benchmark.py \
      --agent dynaq --env $env --seed $seed \
      --total_steps 500000 \
      --save_dir results \
      --eval_generalization

    echo "--- Dreamer | $env | seed $seed ---"
    $PYTHON benchmark.py \
      --agent dreamer --env $env --seed $seed \
      --total_steps 200000 \
      --save_dir results \
      --eval_generalization
  done
done

echo "Phase 1 complete."

# ==============================================================================
# Phase 2 — Ablation A: Dyna-Q planning ratio k (plan.txt §7.1-A)
# Sweep k ∈ {0, 1, 10, 20}. Default k=5 is already in results/logs/.
# Env: doorkey (hardest tractable), seed: 42, steps: 200K
# ==============================================================================
echo ""
echo "========================================"
echo "  Phase 2: Ablation A — Dyna-Q k sweep"
echo "========================================"

for k in 0 1 10 20; do
  echo "--- Dyna-Q k=$k | doorkey | seed 42 ---"
  $PYTHON benchmark.py \
    --agent dynaq --env doorkey --seed 42 \
    --total_steps 200000 \
    --planning_k $k \
    --save_dir results/ablation_k
done

echo "Ablation A complete."

# ==============================================================================
# Phase 3 — Ablation B: Dreamer imagination horizon H (plan.txt §7.1-B)
# Sweep H ∈ {5, 10, 20, 25}. Default H=15 is already in results/logs/.
# Env: multiroom (long credit assignment chain), seed: 42, steps: 200K
# ==============================================================================
echo ""
echo "========================================"
echo "  Phase 3: Ablation B — Dreamer H sweep"
echo "========================================"

for h in 5 10 20 25; do
  echo "--- Dreamer H=$h | multiroom | seed 42 ---"
  $PYTHON benchmark.py \
    --agent dreamer --env multiroom --seed 42 \
    --total_steps 200000 \
    --imagination_h $h \
    --save_dir results/ablation_h
done

echo "Ablation B complete."

# ==============================================================================
# Phase 4 — Visualization (plan.txt §6)
# ==============================================================================
echo ""
echo "========================================"
echo "  Phase 4: Generating all figures"
echo "========================================"

$PYTHON visualize.py --results_dir results/

echo ""
echo "All done. Artifacts:"
echo "  Training logs:      results/logs/"
echo "  Checkpoints:        results/checkpoints/"
echo "  Figures:            results/plots/"
echo "  Generalization CSV: results/tables/generalization_results.csv"
