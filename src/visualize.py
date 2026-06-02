"""
visualize.py — Generate all figures for the EEC 289A final project report.

Usage:
    python visualize.py --results_dir results/
    python visualize.py --results_dir results/ --figures 1,2,3,4,10,11

Figures generated (plan.txt §6):
    1  — Learning curves (4 envs, 3 agents, ±1 std shading)
    2  — Sample efficiency bar chart (steps to 50% and 80% success)
    3  — Wall-clock time bar chart (minutes to 80% success)
    4  — Generalization heatmap (8x8 vs 16x16 zero-shot)
    10 — Ablation: Dyna-Q planning ratio k
    11 — Ablation: Dreamer imagination horizon H
    5,6,7,8,9,12,13 — Stub (require checkpoint rollout infrastructure)

Additional figures (always run, skip gracefully with no data):
    loss_curves_{ppo,dynaq,dreamer}.png — training loss/diagnostic columns per agent
    episode_length_curves.png           — mean episode length, 2×2 grid
    return_curves.png                   — mean episode return, 2×2 grid
    per_seed_variance.png               — individual seed runs vs mean (Empty-8×8)
    grad_norm_diagnostics.png           — gradient norm EMA per agent
    tables/summary_table.csv            — 12-row per-(agent, env) summary table

Outputs saved to: <results_dir>/plots/
"""

import argparse
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    warnings.warn("seaborn not installed — Figure 4 heatmap will use matplotlib fallback.")

# ---------------------------------------------------------------------------
# Style constants (plan.txt §6, Figure 1 spec)
# ---------------------------------------------------------------------------
AGENT_COLORS = {
    "ppo":     "#2196F3",   # blue
    "dynaq":   "#FF9800",   # orange
    "dreamer": "#4CAF50",   # green
}
AGENT_LABELS = {
    "ppo":     "PPO (Model-Free)",
    "dynaq":   "Neural Dyna-Q (Model-Based)",
    "dreamer": "Dreamer-style World Model",
}
ENV_TITLES = {
    "empty":       "Empty-8×8",
    "doorkey":     "DoorKey-8×8",
    "multiroom":   "MultiRoom-N4-S5",
    "keycorridor": "KeyCorridor-S3-R1",
}
STEP_GRID = np.arange(0, 500_001, 1_000)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_filename(fname: str):
    """
    Parse log CSV filename into (agent, env, seed, variant_tag).
    Patterns: {agent}_{env}_seed{N}.csv
              {agent}_{env}_seed{N}_k{K}.csv
              {agent}_{env}_seed{N}_h{H}.csv
    """
    name = fname.replace(".csv", "")
    m = re.match(
        r"^(ppo|dynaq|dreamer)_(\w+)_seed(\d+)(_k\d+|_h\d+)?$", name
    )
    if not m:
        return None
    agent, env, seed, variant = m.group(1), m.group(2), int(m.group(3)), m.group(4) or ""
    return agent, env, seed, variant


def load_logs(results_dir: str):
    """
    Scan results_dir/logs/*.csv and return nested dict:
        logs[agent][env][seed] = pd.DataFrame   (for variant == "")
        ablation_k[k_val]      = pd.DataFrame   (dynaq_doorkey_seed42_k{K}.csv)
        ablation_h[h_val]      = pd.DataFrame   (dreamer_multiroom_seed42_h{H}.csv)
    """
    log_dir = os.path.join(results_dir, "logs")
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    logs = {}
    ablation_k = {}
    ablation_h = {}

    for fname in os.listdir(log_dir):
        if not fname.endswith(".csv"):
            continue
        parsed = _parse_filename(fname)
        if parsed is None:
            continue
        agent, env, seed, variant = parsed
        path = os.path.join(log_dir, fname)
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if variant.startswith("_k"):
            k_val = int(variant[2:])
            ablation_k[k_val] = df
        elif variant.startswith("_h"):
            h_val = int(variant[2:])
            ablation_h[h_val] = df
        else:
            logs.setdefault(agent, {}).setdefault(env, {})[seed] = df

    return logs, ablation_k, ablation_h


def interpolate(df: pd.DataFrame, grid: np.ndarray,
                col: str = "success_rate") -> np.ndarray:
    """Resample a log DataFrame's `col` onto a common step grid."""
    if "step" not in df.columns or col not in df.columns:
        return np.full(len(grid), np.nan)
    steps = df["step"].values.astype(float)
    vals  = df[col].values.astype(float)
    if len(steps) == 0:
        return np.full(len(grid), np.nan)
    # Prepend (0, 0) so interpolation starts at zero
    steps = np.concatenate([[0], steps])
    vals  = np.concatenate([[0], vals])
    return np.interp(grid, steps, vals, left=np.nan, right=vals[-1])


def steps_to_threshold(df: pd.DataFrame, threshold: float,
                        col: str = "success_rate") -> float:
    """Return env steps when success_rate first >= threshold, or np.inf."""
    if "step" not in df.columns or col not in df.columns:
        return np.inf
    mask = df[col] >= threshold
    if not mask.any():
        return np.inf
    return float(df.loc[mask.idxmax(), "step"])


def wallclock_to_threshold(df: pd.DataFrame, threshold: float,
                            col: str = "success_rate") -> float:
    """Return wall-clock seconds when success_rate first >= threshold."""
    if "step" not in df.columns or col not in df.columns or "wall_time_sec" not in df.columns:
        return np.inf
    mask = df[col] >= threshold
    if not mask.any():
        return np.inf
    return float(df.loc[mask.idxmax(), "wall_time_sec"])


# ---------------------------------------------------------------------------
# Figure 1 — Learning curves
# ---------------------------------------------------------------------------

def fig1_learning_curves(logs: dict, save_dir: str):
    envs = ["empty", "doorkey", "multiroom", "keycorridor"]
    agents = ["ppo", "dynaq", "dreamer"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, env in zip(axes.flat, envs):
        for agent in agents:
            seed_dfs = logs.get(agent, {}).get(env, {})
            if not seed_dfs:
                continue
            curves = np.stack([interpolate(df, STEP_GRID) for df in seed_dfs.values()])
            mean = np.nanmean(curves, axis=0)
            std  = np.nanstd(curves, axis=0)
            color = AGENT_COLORS[agent]
            ax.plot(STEP_GRID / 1_000, mean, color=color,
                    linewidth=2.5, label=AGENT_LABELS[agent])
            ax.fill_between(STEP_GRID / 1_000, mean - std, mean + std,
                            alpha=0.15, color=color)

        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_title(ENV_TITLES.get(env, env), fontsize=14, fontweight="bold")
        ax.set_xlabel("Environment Steps (K)", fontsize=11)
        ax.set_ylabel("Success Rate", fontsize=11)
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Also save individual env figure
        fig_single, ax_s = plt.subplots(figsize=(10, 7))
        for agent in agents:
            seed_dfs = logs.get(agent, {}).get(env, {})
            if not seed_dfs:
                continue
            curves = np.stack([interpolate(df, STEP_GRID) for df in seed_dfs.values()])
            mean = np.nanmean(curves, axis=0)
            std  = np.nanstd(curves, axis=0)
            color = AGENT_COLORS[agent]
            ax_s.plot(STEP_GRID / 1_000, mean, color=color,
                      linewidth=2.5, label=AGENT_LABELS[agent])
            ax_s.fill_between(STEP_GRID / 1_000, mean - std, mean + std,
                               alpha=0.15, color=color)
        ax_s.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax_s.axhline(0.8, color="gray", linestyle="--", alpha=0.5)
        ax_s.set_title(ENV_TITLES.get(env, env), fontsize=14, fontweight="bold")
        ax_s.set_xlabel("Environment Steps (K)", fontsize=11)
        ax_s.set_ylabel("Success Rate", fontsize=11)
        ax_s.set_xlim(0, 500)
        ax_s.set_ylim(0, 1.05)
        ax_s.legend(loc="upper left", fontsize=11)
        ax_s.grid(True, alpha=0.3)
        fig_single.tight_layout()
        p = os.path.join(save_dir, f"learning_curves_{env}.png")
        fig_single.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig_single)
        print(f"  Saved {p}")

    fig.suptitle("Learning Curves: PPO vs Neural Dyna-Q vs Dreamer", fontsize=15, y=1.01)
    fig.tight_layout()
    p = os.path.join(save_dir, "learning_curves_all.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ---------------------------------------------------------------------------
# Figure 2 — Sample efficiency bar chart
# ---------------------------------------------------------------------------

def _efficiency_bar_chart(logs: dict, threshold: float, save_dir: str,
                           fname: str, title: str):
    envs   = ["empty", "doorkey", "multiroom", "keycorridor"]
    agents = ["ppo", "dynaq", "dreamer"]
    n_envs   = len(envs)
    n_agents = len(agents)
    bar_w = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_envs)

    for i, agent in enumerate(agents):
        means, errs, never = [], [], []
        for env in envs:
            seed_dfs = logs.get(agent, {}).get(env, {})
            vals = [steps_to_threshold(df, threshold) / 1_000 for df in seed_dfs.values()]
            cap = 500.0  # represent as 500K if never reached
            capped = [min(v, cap) for v in vals]
            hit    = [v < np.inf for v in vals]
            means.append(np.mean(capped) if capped else cap)
            errs.append(np.std(capped) if len(capped) > 1 else 0.0)
            never.append(not any(hit))

        offset = (i - n_agents / 2 + 0.5) * (bar_w + 0.02)
        bars = ax.bar(x + offset, means, bar_w,
                      yerr=errs, capsize=5,
                      color=AGENT_COLORS[agent],
                      label=AGENT_LABELS[agent],
                      alpha=0.85, error_kw={"elinewidth": 1.5})

        for j, (bar, nev) in enumerate(zip(bars, never)):
            h = bar.get_height()
            if nev:
                bar.set_hatch("///")
                bar.set_alpha(0.5)
                ax.text(bar.get_x() + bar.get_width() / 2, h + 5,
                        "N/R", ha="center", va="bottom", fontsize=8, color="gray")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                        f"{h:.0f}K", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([ENV_TITLES.get(e, e) for e in envs], fontsize=11)
    ax.set_ylabel("Steps to Threshold (K)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    p = os.path.join(save_dir, fname)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


def fig2_sample_efficiency(logs: dict, save_dir: str):
    _efficiency_bar_chart(logs, 0.5, save_dir,
                          "sample_efficiency_50pct.png",
                          "Sample Efficiency — Steps to 50% Success Rate")
    _efficiency_bar_chart(logs, 0.8, save_dir,
                          "sample_efficiency_80pct.png",
                          "Sample Efficiency — Steps to 80% Success Rate")


# ---------------------------------------------------------------------------
# Figure 3 — Wall-clock time bar chart
# ---------------------------------------------------------------------------

def fig3_wallclock(logs: dict, save_dir: str):
    envs   = ["empty", "doorkey", "multiroom", "keycorridor"]
    agents = ["ppo", "dynaq", "dreamer"]
    n_envs   = len(envs)
    n_agents = len(agents)
    bar_w = 0.25
    threshold = 0.8

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_envs)

    for i, agent in enumerate(agents):
        means, errs, never = [], [], []
        for env in envs:
            seed_dfs = logs.get(agent, {}).get(env, {})
            vals = [wallclock_to_threshold(df, threshold) / 60.0
                    for df in seed_dfs.values()]   # convert to minutes
            cap = 600.0
            capped = [min(v, cap) for v in vals]
            hit    = [v < np.inf for v in vals]
            means.append(np.mean(capped) if capped else cap)
            errs.append(np.std(capped) if len(capped) > 1 else 0.0)
            never.append(not any(hit))

        offset = (i - n_agents / 2 + 0.5) * (bar_w + 0.02)
        bars = ax.bar(x + offset, means, bar_w,
                      yerr=errs, capsize=5,
                      color=AGENT_COLORS[agent],
                      label=AGENT_LABELS[agent],
                      alpha=0.85, error_kw={"elinewidth": 1.5})

        for bar, nev in zip(bars, never):
            h = bar.get_height()
            if nev:
                bar.set_hatch("///")
                bar.set_alpha(0.5)
                ax.text(bar.get_x() + bar.get_width() / 2, h + 5,
                        "N/R", ha="center", va="bottom", fontsize=8, color="gray")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                        f"{h:.0f}m", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([ENV_TITLES.get(e, e) for e in envs], fontsize=11)
    ax.set_ylabel("Wall-Clock Minutes to 80% Success", fontsize=11)
    ax.set_title("Wall-Clock Time Comparison (to 80% Success Rate)", fontsize=13,
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    p = os.path.join(save_dir, "wallclock.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ---------------------------------------------------------------------------
# Figure 4 — Generalization heatmap
# ---------------------------------------------------------------------------

def fig4_generalization_heatmap(results_dir: str, logs: dict, save_dir: str):
    gen_path = os.path.join(results_dir, "tables", "generalization_results.csv")
    if not os.path.exists(gen_path):
        print(f"  [Figure 4] generalization_results.csv not found at {gen_path} — skipping.")
        print("  Run benchmark.py with --eval_generalization to generate it.")
        return

    gen_df = pd.read_csv(gen_path)

    envs   = ["empty", "doorkey", "multiroom", "keycorridor"]
    agents = ["ppo", "dynaq", "dreamer"]

    # Build matrices
    gen_mat   = np.full((len(agents), len(envs)), np.nan)
    train_mat = np.full((len(agents), len(envs)), np.nan)
    delta_mat = np.full((len(agents), len(envs)), np.nan)

    for i, agent in enumerate(agents):
        for j, env in enumerate(envs):
            # Generalization: mean across seeds from gen_df
            mask = (gen_df["agent"] == agent) & (gen_df["env"] == env)
            if mask.any():
                gen_mat[i, j] = gen_df.loc[mask, "gen_success_rate_16x16"].mean()

            # Training success: mean of final-step success_rate across seeds
            seed_dfs = logs.get(agent, {}).get(env, {})
            finals = []
            for df in seed_dfs.values():
                if "success_rate" in df.columns and len(df) > 0:
                    finals.append(df["success_rate"].iloc[-1])
            if finals:
                train_mat[i, j] = np.mean(finals)

            if not np.isnan(gen_mat[i, j]) and not np.isnan(train_mat[i, j]):
                delta_mat[i, j] = train_mat[i, j] - gen_mat[i, j]

    row_labels = [AGENT_LABELS[a].split(" (")[0] for a in agents]
    col_labels = [ENV_TITLES.get(e, e) for e in envs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    def _heatmap(ax, mat, title, cmap, vmin=0, vmax=1):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(col_labels)))
        ax.set_yticks(range(len(row_labels)))
        ax.set_xticklabels(col_labels, fontsize=10, rotation=20, ha="right")
        ax.set_yticklabels(row_labels, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for ii in range(len(row_labels)):
            for jj in range(len(col_labels)):
                val = mat[ii, jj]
                txt = f"{val:.2f}" if not np.isnan(val) else "—"
                ax.text(jj, ii, txt, ha="center", va="center",
                        fontsize=11, color="black")

    _heatmap(ax1, gen_mat, "Zero-Shot Success Rate (16×16)", "RdYlGn")
    _heatmap(ax2, delta_mat, "Success Drop: 8×8 − 16×16", "RdYlGn_r", vmin=0, vmax=1)

    fig.suptitle("Zero-Shot Generalization to Larger Environments", fontsize=14, y=1.02)
    fig.tight_layout()
    p = os.path.join(save_dir, "generalization_heatmap.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ---------------------------------------------------------------------------
# Figure 10 — Ablation: Dyna-Q planning ratio k
# ---------------------------------------------------------------------------

def fig10_ablation_k(results_dir: str, save_dir: str):
    ablation_dir = os.path.join(results_dir, "ablation_k", "logs")
    if not os.path.isdir(ablation_dir):
        print(f"  [Figure 10] {ablation_dir} not found — skipping.")
        return

    k_files = {}
    for fname in os.listdir(ablation_dir):
        if not fname.endswith(".csv"):
            continue
        m = re.search(r"_k(\d+)\.csv$", fname)
        if m:
            k_files[int(m.group(1))] = os.path.join(ablation_dir, fname)

    # Also include default k=5 from main results (doorkey, seed 42)
    default_path = os.path.join(results_dir, "logs", "dynaq_doorkey_seed42.csv")
    if os.path.exists(default_path) and 5 not in k_files:
        k_files[5] = default_path

    if not k_files:
        print("  [Figure 10] No ablation_k log files found — skipping.")
        return

    cmap = plt.cm.viridis
    k_vals = sorted(k_files.keys())
    colors = [cmap(i / max(len(k_vals) - 1, 1)) for i in range(len(k_vals))]

    fig, ax = plt.subplots(figsize=(10, 7))
    threshold = 0.8

    for color, k in zip(colors, k_vals):
        df = pd.read_csv(k_files[k])
        curve = interpolate(df, STEP_GRID)
        ax.plot(STEP_GRID / 1_000, curve, color=color, linewidth=2.5,
                label=f"k = {k}" + (" (DQN baseline)" if k == 0 else ""))
        # Vertical marker at first 80% step
        t = steps_to_threshold(df, threshold)
        if t < np.inf:
            ax.axvline(t / 1_000, color=color, linestyle=":", alpha=0.6, linewidth=1.2)

    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Environment Steps (K)", fontsize=11)
    ax.set_ylabel("Success Rate", fontsize=11)
    ax.set_title("Ablation A — Dyna-Q Planning Ratio (DoorKey-8×8, seed 42)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(save_dir, "ablation_planning_ratio.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ---------------------------------------------------------------------------
# Figure 11 — Ablation: Dreamer imagination horizon H
# ---------------------------------------------------------------------------

def fig11_ablation_h(results_dir: str, save_dir: str):
    ablation_dir = os.path.join(results_dir, "ablation_h", "logs")
    if not os.path.isdir(ablation_dir):
        print(f"  [Figure 11] {ablation_dir} not found — skipping.")
        return

    h_files = {}
    for fname in os.listdir(ablation_dir):
        if not fname.endswith(".csv"):
            continue
        m = re.search(r"_h(\d+)\.csv$", fname)
        if m:
            h_files[int(m.group(1))] = os.path.join(ablation_dir, fname)

    default_path = os.path.join(results_dir, "logs", "dreamer_multiroom_seed42.csv")
    if os.path.exists(default_path) and 15 not in h_files:
        h_files[15] = default_path

    if not h_files:
        print("  [Figure 11] No ablation_h log files found — skipping.")
        return

    cmap = plt.cm.plasma
    h_vals = sorted(h_files.keys())
    colors = [cmap(i / max(len(h_vals) - 1, 1)) for i in range(len(h_vals))]

    fig, ax = plt.subplots(figsize=(10, 7))
    for color, h in zip(colors, h_vals):
        df = pd.read_csv(h_files[h])
        curve = interpolate(df, STEP_GRID)
        ax.plot(STEP_GRID / 1_000, curve, color=color, linewidth=2.5, label=f"H = {h}")
        t = steps_to_threshold(df, 0.8)
        if t < np.inf:
            ax.axvline(t / 1_000, color=color, linestyle=":", alpha=0.6, linewidth=1.2)

    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Environment Steps (K)", fontsize=11)
    ax.set_ylabel("Success Rate", fontsize=11)
    ax.set_title("Ablation B — Dreamer Imagination Horizon (MultiRoom-N4-S5, seed 42)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(save_dir, "ablation_horizon.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ---------------------------------------------------------------------------
# Figure: Loss diagnostic curves
# ---------------------------------------------------------------------------

LOSS_COLS = {
    "ppo":     ["policy_loss", "value_loss", "entropy"],
    "dynaq":   ["q_loss", "model_loss", "epsilon"],
    "dreamer": ["wm_loss", "recon_loss", "kl_loss", "actor_loss", "critic_loss"],
}
LOSS_LABELS = {
    "policy_loss": "Policy Loss", "value_loss": "Value Loss", "entropy": "Entropy",
    "q_loss": "Q-Network Loss", "model_loss": "Model Loss", "epsilon": "ε (Exploration)",
    "wm_loss": "World Model Loss", "recon_loss": "Recon Loss", "kl_loss": "KL Loss",
    "actor_loss": "Actor Loss", "critic_loss": "Critic Loss",
}


def fig_loss_curves(logs: dict, save_dir: str):
    """One figure per agent — loss/diagnostic columns over training steps."""
    envs = ["empty", "doorkey", "multiroom", "keycorridor"]
    env_colors = ["#E91E63", "#9C27B0", "#00BCD4", "#FF5722"]

    for agent, cols in LOSS_COLS.items():
        # Filter to columns that actually exist in any loaded data
        avail_cols = []
        for col in cols:
            for env in envs:
                for df in logs.get(agent, {}).get(env, {}).values():
                    if col in df.columns:
                        avail_cols.append(col)
                        break
                else:
                    continue
                break

        if not avail_cols:
            print(f"  [Loss curves {agent}] No loss columns found — skipping.")
            continue

        n_cols = len(avail_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        if n_cols == 1:
            axes = [axes]

        for ax, col in zip(axes, avail_cols):
            has_data = False
            for env, ec in zip(envs, env_colors):
                seed_dfs = logs.get(agent, {}).get(env, {})
                if not seed_dfs:
                    continue
                curves = []
                for df in seed_dfs.values():
                    if col in df.columns and "step" in df.columns and len(df) > 0:
                        curves.append(np.interp(STEP_GRID,
                                                 df["step"].values.astype(float),
                                                 df[col].values.astype(float),
                                                 left=np.nan, right=df[col].iloc[-1]))
                if not curves:
                    continue
                has_data = True
                arr = np.array(curves)
                mean = np.nanmean(arr, axis=0)
                std  = np.nanstd(arr, axis=0)
                ax.plot(STEP_GRID / 1_000, mean, color=ec, linewidth=2,
                        label=ENV_TITLES.get(env, env))
                ax.fill_between(STEP_GRID / 1_000, mean - std, mean + std,
                                alpha=0.15, color=ec)

            ax.set_title(LOSS_LABELS.get(col, col), fontsize=12, fontweight="bold")
            ax.set_xlabel("Steps (K)", fontsize=10)
            ax.set_ylabel(LOSS_LABELS.get(col, col), fontsize=10)
            ax.grid(True, alpha=0.3)
            if has_data:
                ax.legend(fontsize=8)

        fig.suptitle(f"Loss Diagnostics — {AGENT_LABELS.get(agent, agent)}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        p = os.path.join(save_dir, f"loss_curves_{agent}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {p}")


# ---------------------------------------------------------------------------
# Figure: Episode length & mean return curves
# ---------------------------------------------------------------------------

def _plot_metric_2x2(logs: dict, col: str, ylabel: str, title: str,
                     fname: str, save_dir: str):
    """2×2 grid across envs, like Fig 1, for any scalar metric column."""
    envs = ["empty", "doorkey", "multiroom", "keycorridor"]
    agents = ["ppo", "dynaq", "dreamer"]
    has_any = any(
        col in df.columns
        for agent in agents
        for env in envs
        for df in logs.get(agent, {}).get(env, {}).values()
    )
    if not has_any:
        print(f"  [{fname}] Column '{col}' not found in any log — skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, env in zip(axes.flat, envs):
        for agent in agents:
            seed_dfs = logs.get(agent, {}).get(env, {})
            curves = []
            for df in seed_dfs.values():
                if col in df.columns and "step" in df.columns and len(df) > 0:
                    curves.append(np.interp(STEP_GRID,
                                             df["step"].values.astype(float),
                                             df[col].values.astype(float),
                                             left=np.nan, right=df[col].iloc[-1]))
            if not curves:
                continue
            arr = np.array(curves)
            mean = np.nanmean(arr, axis=0)
            std  = np.nanstd(arr, axis=0)
            color = AGENT_COLORS[agent]
            ax.plot(STEP_GRID / 1_000, mean, color=color, linewidth=2.5,
                    label=AGENT_LABELS[agent])
            ax.fill_between(STEP_GRID / 1_000, mean - std, mean + std,
                            alpha=0.15, color=color)

        ax.set_title(ENV_TITLES.get(env, env), fontsize=14, fontweight="bold")
        ax.set_xlabel("Environment Steps (K)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlim(0, 500)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=15, y=1.01)
    fig.tight_layout()
    p = os.path.join(save_dir, fname)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


def fig_episode_length(logs: dict, save_dir: str):
    _plot_metric_2x2(logs, "mean_ep_length", "Mean Episode Length (steps)",
                     "Episode Length — Shorter = Faster Goal Reaching",
                     "episode_length_curves.png", save_dir)


def fig_mean_return(logs: dict, save_dir: str):
    _plot_metric_2x2(logs, "mean_return", "Mean Episode Return",
                     "Mean Return Curves — PPO vs Dyna-Q vs Dreamer",
                     "return_curves.png", save_dir)


# ---------------------------------------------------------------------------
# Figure: Per-seed variance (training stability)
# ---------------------------------------------------------------------------

def fig_per_seed_variance(logs: dict, save_dir: str):
    """For 'empty' env, show individual seed curves behind the mean — one panel per agent."""
    env = "empty"
    agents = ["ppo", "dynaq", "dreamer"]

    any_data = any(logs.get(a, {}).get(env, {}) for a in agents)
    if not any_data:
        print("  [per_seed_variance] No data for 'empty' env — skipping.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, agent in zip(axes, agents):
        seed_dfs = logs.get(agent, {}).get(env, {})
        color = AGENT_COLORS[agent]
        all_curves = []
        for seed, df in seed_dfs.items():
            curve = interpolate(df, STEP_GRID)
            all_curves.append(curve)
            ax.plot(STEP_GRID / 1_000, curve, color=color, linewidth=1.2, alpha=0.4)

        if all_curves:
            mean = np.nanmean(np.array(all_curves), axis=0)
            ax.plot(STEP_GRID / 1_000, mean, color=color, linewidth=3,
                    label=f"Mean (n={len(all_curves)})")

        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
        ax.axhline(0.8, color="gray", linestyle="--", alpha=0.4, linewidth=1)
        ax.set_title(AGENT_LABELS.get(agent, agent), fontsize=12, fontweight="bold")
        ax.set_xlabel("Steps (K)", fontsize=10)
        ax.set_ylabel("Success Rate", fontsize=10)
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Seed Variance — Empty-8×8 (Training Stability)", fontsize=14, y=1.01)
    fig.tight_layout()
    p = os.path.join(save_dir, "per_seed_variance.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ---------------------------------------------------------------------------
# Figure: Gradient norm diagnostics
# ---------------------------------------------------------------------------

def fig_grad_norm(logs: dict, save_dir: str):
    """3 subplots (one per agent) — grad_norm EMA over training steps, lines per seed."""
    col = "grad_norm"
    agents = ["ppo", "dynaq", "dreamer"]
    envs = ["empty", "doorkey", "multiroom", "keycorridor"]
    env_colors = ["#E91E63", "#9C27B0", "#00BCD4", "#FF5722"]

    has_any = any(
        col in df.columns
        for agent in agents
        for env in envs
        for df in logs.get(agent, {}).get(env, {}).values()
    )
    if not has_any:
        print("  [grad_norm] Column 'grad_norm' not present in any log — "
              "run benchmark.py after this session to populate it.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, agent in zip(axes, agents):
        has_data = False
        for env, ec in zip(envs, env_colors):
            env_label = ENV_TITLES.get(env, env)
            first_seed = True
            for df in logs.get(agent, {}).get(env, {}).values():
                if col not in df.columns or "step" not in df.columns:
                    continue
                has_data = True
                # Only the first seed line carries the legend label for this env.
                ax.plot(df["step"].values / 1_000, df[col].values,
                        color=ec, linewidth=1.5, alpha=0.7,
                        label=env_label if first_seed else "_nolegend_")
                first_seed = False

        ax.set_title(AGENT_LABELS.get(agent, agent), fontsize=11, fontweight="bold")
        ax.set_xlabel("Steps (K)", fontsize=10)
        ax.set_ylabel("Gradient Norm (EMA)", fontsize=10)
        ax.grid(True, alpha=0.3)
        if has_data:
            ax.legend(fontsize=8)

    fig.suptitle("Gradient Norm Diagnostics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(save_dir, "grad_norm_diagnostics.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ---------------------------------------------------------------------------
# Figure: Comprehensive summary table
# ---------------------------------------------------------------------------

def fig_summary_table(logs: dict, results_dir: str, save_dir: str):
    """Build 12-row summary table and save to results/tables/summary_table.csv."""
    envs   = ["empty", "doorkey", "multiroom", "keycorridor"]
    agents = ["ppo", "dynaq", "dreamer"]

    # Load generalization results if available
    gen_path = os.path.join(results_dir, "tables", "generalization_results.csv")
    gen_df = None
    if os.path.exists(gen_path):
        try:
            gen_df = pd.read_csv(gen_path)
        except Exception:
            pass

    rows = []
    for agent in agents:
        for env in envs:
            seed_dfs = logs.get(agent, {}).get(env, {})
            if not seed_dfs:
                rows.append({
                    "agent": agent, "env": env,
                    "final_sr_mean": float("nan"), "final_sr_std": float("nan"),
                    "steps_to_50K": float("nan"), "steps_to_80K": float("nan"),
                    "wallclock_80_min": float("nan"), "gen_rate_16x16": float("nan"),
                })
                continue

            finals = []
            s50s, s80s, wc80s = [], [], []
            for df in seed_dfs.values():
                if "success_rate" in df.columns and len(df):
                    finals.append(df["success_rate"].iloc[-1])
                s50s.append(steps_to_threshold(df, 0.5) / 1_000)
                s80s.append(steps_to_threshold(df, 0.8) / 1_000)
                wc80s.append(wallclock_to_threshold(df, 0.8) / 60.0)

            # Replace inf with nan for table output
            def _clean(v):
                return float("nan") if v == np.inf or v == float("inf") else v

            gen_rate = float("nan")
            if gen_df is not None:
                mask = (gen_df["agent"] == agent) & (gen_df["env"] == env)
                if mask.any():
                    gen_rate = gen_df.loc[mask, "gen_success_rate_16x16"].mean()

            rows.append({
                "agent": agent,
                "env": env,
                "final_sr_mean": round(np.mean(finals), 3) if finals else float("nan"),
                "final_sr_std":  round(np.std(finals),  3) if finals else float("nan"),
                "steps_to_50K":  round(np.nanmean([_clean(v) for v in s50s]), 1),
                "steps_to_80K":  round(np.nanmean([_clean(v) for v in s80s]), 1),
                "wallclock_80_min": round(np.nanmean([_clean(v) for v in wc80s]), 1),
                "gen_rate_16x16": round(gen_rate, 3) if np.isfinite(gen_rate) else float("nan"),
            })

    table = pd.DataFrame(rows)
    tables_dir = os.path.join(results_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    csv_path = os.path.join(tables_dir, "summary_table.csv")
    table.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    # Pretty-print to console
    print("\n  ── Summary Table ──")
    print(table.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Stubs for figures requiring checkpoint rollout infrastructure
# ---------------------------------------------------------------------------

def _stub(fig_num: int, description: str, save_dir: str):
    print(f"  [Figure {fig_num}] STUB — {description}")
    print(f"             Requires checkpoint rollout. Skipping.")


def fig5_model_accuracy(save_dir: str):
    _stub(5, "World model reward/latent prediction accuracy (needs checkpoint rollout)", save_dir)


def fig6_trajectory_comparison(save_dir: str):
    _stub(6, "Real vs imagined trajectory frames (needs env.render + checkpoint)", save_dir)


def fig7_policy_gifs(save_dir: str):
    _stub(7, "Policy evolution GIFs (needs env.render + multi-checkpoint loading)", save_dir)


def fig8_visitation_heatmaps(save_dir: str):
    _stub(8, "Agent visitation heatmaps (needs position tracking during training/eval)", save_dir)


def fig9_failure_gallery(save_dir: str):
    _stub(9, "Failure mode gallery (needs env.render + checkpoint rollout)", save_dir)


def fig12_credit_assignment(save_dir: str):
    _stub(12, "Credit assignment — key pickup analysis (needs DoorKey position tracking)", save_dir)


def fig13_state_coverage(save_dir: str):
    _stub(13, "State coverage over training (needs position tracking during training)", save_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate all report figures from benchmark training logs.")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Root results directory (contains logs/, tables/, plots/).")
    parser.add_argument("--figures", type=str, default="all",
                        help="Comma-separated list of figure numbers to generate, e.g. '1,2,4'. "
                             "Use 'all' for all available figures.")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir
    plots_dir   = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if args.figures == "all":
        wanted = set(range(1, 14))
    else:
        try:
            wanted = set(int(x.strip()) for x in args.figures.split(","))
        except ValueError:
            print(f"Invalid --figures value: {args.figures}")
            return

    print(f"Loading logs from {results_dir}/logs/ ...")
    try:
        logs, ablation_k, ablation_h = load_logs(results_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    n_runs = sum(len(seeds) for agent in logs.values() for seeds in agent.values())
    print(f"  Found {n_runs} training runs across {len(logs)} agents.")

    if 1 in wanted:
        print("\n[Figure 1] Learning curves...")
        fig1_learning_curves(logs, plots_dir)

    if 2 in wanted:
        print("\n[Figure 2] Sample efficiency bar charts...")
        fig2_sample_efficiency(logs, plots_dir)

    if 3 in wanted:
        print("\n[Figure 3] Wall-clock time bar chart...")
        fig3_wallclock(logs, plots_dir)

    if 4 in wanted:
        print("\n[Figure 4] Generalization heatmap...")
        fig4_generalization_heatmap(results_dir, logs, plots_dir)

    if 5 in wanted:
        fig5_model_accuracy(plots_dir)
    if 6 in wanted:
        fig6_trajectory_comparison(plots_dir)
    if 7 in wanted:
        fig7_policy_gifs(plots_dir)
    if 8 in wanted:
        fig8_visitation_heatmaps(plots_dir)
    if 9 in wanted:
        fig9_failure_gallery(plots_dir)

    if 10 in wanted:
        print("\n[Figure 10] Ablation: Dyna-Q planning ratio...")
        fig10_ablation_k(results_dir, plots_dir)

    if 11 in wanted:
        print("\n[Figure 11] Ablation: Dreamer imagination horizon...")
        fig11_ablation_h(results_dir, plots_dir)

    if 12 in wanted:
        fig12_credit_assignment(plots_dir)
    if 13 in wanted:
        fig13_state_coverage(plots_dir)

    print("\n[Loss curves] Loss diagnostic figures...")
    fig_loss_curves(logs, plots_dir)

    print("\n[Episode length] Episode length curves...")
    fig_episode_length(logs, plots_dir)

    print("\n[Return curves] Mean return curves...")
    fig_mean_return(logs, plots_dir)

    print("\n[Per-seed variance] Training stability figure...")
    fig_per_seed_variance(logs, plots_dir)

    print("\n[Grad norm] Gradient norm diagnostics...")
    fig_grad_norm(logs, plots_dir)

    print("\n[Summary table] Building comprehensive summary table...")
    fig_summary_table(logs, results_dir, plots_dir)

    print(f"\nDone. Plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
