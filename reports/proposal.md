# EEC 289A — Final Project Proposal

**Title:** Sample Efficiency in Sparse-Reward Navigation: A Comparative Study of Model-Free and Model-Based Reinforcement Learning

**Course:** EEC 289A — Sensorimotor Learning  
**Institution:** University of California, Davis  
**Date:** May 20, 2026

**Team Members:**
- Pranav Manimaran
- Vandana Mansur

---

## 1. Motivation

A central challenge in deploying reinforcement learning (RL) to real-world systems — such as robotics, autonomous navigation, and healthcare — is *sample efficiency*: the number of environment interactions required to learn a competent policy. In physical systems, each interaction carries a time and cost penalty, making it impractical to rely on algorithms that require millions of environment steps.

The fundamental divide in RL methodology lies between **model-free** approaches, which learn policies directly from experience, and **model-based** approaches, which additionally learn a predictive model of the environment to generate synthetic experience and plan ahead. Despite substantial theoretical motivation for model-based methods, their practical advantages — and failure modes — remain context-dependent and undercharacterized, particularly in sparse-reward settings where the reward signal provides minimal learning guidance.

This project performs a rigorous, empirical comparison of three representative algorithms across a suite of sparse-reward navigation tasks, with the goal of identifying when and why model-based RL outperforms its model-free counterpart.

---

## 2. Methods

We evaluate three algorithms spanning the model-free to model-based spectrum:

**PPO (Proximal Policy Optimization)** serves as the model-free baseline. PPO learns a stochastic policy directly from environment rollouts using clipped surrogate objectives, with no explicit representation of environment dynamics. It is well-established, stable, and broadly applicable, but requires substantial interaction data — a limitation that becomes acute under sparse rewards.

**Dyna-Q** represents a classical model-based approach. In addition to policy learning, Dyna-Q maintains a tabular or parametric transition model (*s*, *a*) → (*s*′, *r*) learned from real experience. This model is used to synthesize additional training transitions ("imagined experience"), increasing effective sample utilization without additional environment interaction.

**World Model + Planning** (following the Dreamer / MuZero paradigm) constitutes the most sophisticated approach. A neural encoder compresses raw observations into a compact latent representation; a learned latent dynamics model then predicts future states and rewards entirely within this compressed space. At each decision step, the agent simulates candidate action sequences in latent space and selects the action leading to the highest predicted cumulative reward. This enables multi-step lookahead planning without access to the true environment simulator.

---

## 3. Experimental Setup

All experiments are conducted on **MiniGrid**, a family of lightweight, partially observable grid-world environments well-suited to studying sparse rewards and credit assignment. We evaluate on four tasks of increasing complexity:

| Environment | Description |
|---|---|
| `Empty-8×8` | Navigate to goal in open room (baseline difficulty) |
| `DoorKey-8×8` | Retrieve a key, unlock a door, reach the goal |
| `MultiRoom-N4-S5` | Traverse four connected rooms to reach the goal |
| `KeyCorridor` | Long corridor with multiple locked doors requiring keys |

These environments share a common structure: reward is sparse (+1 only upon goal completion, 0 otherwise), requiring agents to discover temporally extended action sequences with delayed credit assignment.

---

## 4. Evaluation Metrics

We assess each algorithm along four axes:

1. **Sample Efficiency** — Learning curves (success rate vs. environment steps), measuring how quickly each method reaches target performance thresholds (e.g., 50%, 80% success rate).

2. **Computational Cost** — Wall-clock time and approximate FLOPs to reach 80% success, quantifying the real-time cost of planning overhead in model-based methods.

3. **Generalization** — Policies trained on 8×8 grids are evaluated zero-shot on 16×16 variants, probing whether learned world models encode transferable structure.

4. **Failure Mode Analysis** — Qualitative and quantitative analysis of where each method breaks down: random exploration failures in PPO, model hallucination in Dyna-Q, and planning horizon errors in the world model.

---

## 5. Expected Contributions and Outcomes

We hypothesize that:

- The **World Model + Planning** agent will achieve the highest sample efficiency, reaching 80% success with 40–60% fewer environment steps than PPO on tasks requiring multi-step credit assignment (DoorKey, KeyCorridor).
- **Dyna-Q** will occupy a middle ground — more sample efficient than PPO but less capable of long-horizon planning.
- Model-based methods will incur a wall-clock overhead of 1.5–2× relative to PPO, making the benefit task- and budget-dependent.
- World model generalization to larger grids will be meaningfully stronger than PPO, reflecting the acquisition of reusable environmental knowledge.

Beyond numerical results, we aim to produce a structured failure-case analysis and a practical decision framework: *when should a practitioner prefer model-based RL?*

---

