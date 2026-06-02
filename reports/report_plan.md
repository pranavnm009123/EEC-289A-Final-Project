# Final Report Plan — EEC 289A

**Title:** Sample Efficiency in Sparse-Reward Navigation: A Comparative Study of Model-Free and Model-Based Reinforcement Learning  
**Authors:** Pranav Manimaran, Vandana Mansur  
**Course:** EEC 289A — Sensorimotor Learning, UC Davis  
**Due:** June 11, 2026 at 11:59 PM PST  
**Format:** Single PDF, ≤ 8 pages, LaTeX (IEEE two-column style)

---

## Trigger

Write the report **after** the final 36-run results zip arrives from Kaggle. Extract plots to `Reports/final_report/figures/` and read `summary_table.csv` for all numerical results before writing.

---

## File Structure

```
Reports/
  final_report/
    main.tex           ← full LaTeX source
    references.bib     ← BibTeX citations
    figures/           ← all PNG plots from results zip
    Makefile           ← pdflatex + bibtex build
  report_plan.md       ← this file
```

**Build command:**
```
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
**Page check:** `pdfinfo main.pdf | grep Pages` — must be ≤ 8.

---

## Page Budget

| Section | Target Pages |
|---------|-------------|
| Title + Abstract | 0.5 |
| 1. Introduction & Motivation | 0.75 |
| 2. Background & Related Work | 0.5 |
| 3. Methods | 1.5 |
| 4. Experimental Setup | 0.5 |
| 5. Results | 2.25 |
| 6. Discussion & Failure Analysis | 1.0 |
| 7. Conclusion + Contributions | 0.5 |
| References | 0.5 |
| **Total** | **8.0** |

---

## Section-by-Section Content

### Title Block
```
\title{Sample Efficiency in Sparse-Reward Navigation:\\
A Comparative Study of Model-Free and Model-Based Reinforcement Learning}
\author{Pranav Manimaran \and Vandana Mansur\\
EEC 289A --- Sensorimotor Learning, UC Davis, Spring 2026}
```

---

### Abstract (5–7 sentences)

1. Sentence 1 — Problem: sample efficiency under sparse rewards is a central bottleneck for RL in real-world systems.
2. Sentence 2 — Approach: we benchmark three algorithms (PPO, Neural Dyna-Q, Dreamer) across four MiniGrid sparse-reward navigation tasks.
3. Sentence 3 — Setup: 3 seeds × 4 environments × 3 agents = 36 runs; metrics are steps-to-50%/80% success, wall-clock time, zero-shot generalization to 16×16 grids.
4. Sentence 4 — Key finding 1: [fill from results — expected: Dreamer achieves highest sample efficiency on long-horizon tasks (DoorKey, MultiRoom, KeyCorridor)].
5. Sentence 5 — Key finding 2: [fill — expected: Dyna-Q occupies middle ground; PPO fastest wall-clock on simple tasks].
6. Sentence 6 — Takeaway: model-based methods are preferable when interaction cost dominates; PPO remains competitive when compute budget is tight.

---

### §1 Introduction & Motivation (~0.75 page)

**Opening paragraph** — RL deployment in physical systems (robotics, autonomous navigation, healthcare) is bottlenecked by sample efficiency: each real interaction carries time and cost penalties. Sparse rewards exacerbate this — the agent receives no learning signal until the goal is reached, forcing it to discover long action sequences through mostly random exploration.

**Second paragraph** — The central divide in RL: model-free methods (learn policy directly from experience) vs model-based methods (additionally learn a predictive model of the environment to generate synthetic experience and plan ahead). Despite theoretical motivation for model-based methods, practical advantages remain context-dependent and undercharacterized in sparse-reward settings.

**Third paragraph** — Three research questions this paper answers:
1. Which algorithm is most sample-efficient under sparse rewards, and how does the gap vary with task horizon?
2. What is the wall-clock cost of planning overhead, and is it worth the sample efficiency gain?
3. Does a learned world model encode transferable environmental structure that generalizes to larger grids?

---

### §2 Background & Related Work (~0.5 page)

**PPO** — Schulman et al. (2017). Proximal Policy Optimization. Clipped surrogate objective prevents destructive policy updates. Standard model-free baseline.

**Dyna-Q** — Sutton (1990). Integrated architecture: Q-learning augmented with a learned environment model used for planning (simulated experience). Classical model-based approach.

**Dreamer / RSSM** — Hafner et al. (2020). DreamerV2. Recurrent State Space Model learns a compact latent world model; actor-critic trained entirely on imagined rollouts in latent space. Represents the frontier of world-model-based RL.

**MiniGrid** — Chevalier-Boisvert et al. (2023). Lightweight grid-world benchmark designed for sparse-reward and credit-assignment studies.

---

### §3 Methods (~1.5 pages)

#### 3.1 Shared Components
- Observation encoder: CNN (3 conv layers, output dim 256) shared across all agents
- Discount γ = 0.99
- All agents observe full 7×7×3 agent-centric view (fully observable after flattening)

#### 3.2 PPO
- Clipped surrogate objective, clip ε = 0.2
- GAE advantage estimation, λ = 0.95
- Shared CNN → linear → policy head + value head
- Learning rate: 3×10⁻⁴, minibatch size 256, 4 epochs per rollout
- Total environment steps: 500K

#### 3.3 Neural Dyna-Q
- Online Q-network + target Q-network (DQN-style); CNN encoder shared with transition model
- Transition model: MLP predicts (s', r) from (s, a)
- Planning ratio k = 5: for each real step, 5 additional Q-updates using model-generated transitions
- Q-network lr = 1×10⁻⁴, model lr = 1×10⁻³
- ε-greedy exploration: ε decays 1.0 → 0.05 over first 100K steps
- Total environment steps: 500K

#### 3.4 Dreamer (RSSM-based)
- Encoder: CNN → stochastic latent z (32 categories × 32 classes, straight-through gradient)
- Recurrent state: GRU with hidden dim 512
- World model trained on real experience: reconstruction loss + KL loss (free bits = 1.0)
- Actor-critic trained on H = 15 step imagined rollouts (λ-returns, λ = 0.95)
- World model lr = 6×10⁻⁴; actor/critic lr = 8×10⁻⁵
- Total environment steps: 200K (model-based efficiency requires fewer real steps)

#### Hyperparameter Summary Table

| Parameter | PPO | Dyna-Q | Dreamer |
|-----------|-----|--------|---------|
| CNN encoder output | 256 | 256 | 256 |
| Learning rate | 3e-4 | Q: 1e-4, M: 1e-3 | WM: 6e-4, AC: 8e-5 |
| Total steps | 500K | 500K | 200K |
| γ | 0.99 | 0.99 | 0.99 |
| λ | 0.95 (GAE) | — | 0.95 (λ-return) |
| Special | clip ε=0.2 | k=5 planning | H=15 horizon, GRU h=512 |

---

### §4 Experimental Setup (~0.5 page)

#### Environments

| Environment | Key Challenge |
|-------------|--------------|
| MiniGrid-Empty-8x8-v0 | Baseline — open room, short horizon |
| MiniGrid-DoorKey-8x8-v0 | Two-stage: pick up key → unlock door → reach goal |
| MiniGrid-MultiRoom-N4-S5-v0 | Traverse 4 connected rooms, long credit assignment |
| MiniGrid-KeyCorridor-S3-R1-v0 | Longest horizon, multiple locked doors |

All environments: sparse reward (+1 at goal only, 0 otherwise).

#### Protocol
- Seeds: 42, 123, 777 → 3 runs per (agent, environment) pair = **36 total runs**
- Evaluation every 2,048 steps: 100 deterministic rollouts, record success rate
- Hardware: Kaggle 2×NVIDIA T4 GPU (16 GB each), parallel training
- Zero-shot generalization: after training on 8×8, evaluate on 16×16 variant (no fine-tuning)

---

### §5 Results (~2.25 pages)

**Subsection order:**
1. Learning curves (main figure — full width, 2×2 grid)
2. Sample efficiency (steps-to-threshold bar charts, side by side)
3. Wall-clock cost
4. Zero-shot generalization

**Writing approach:** State each result as a numbered observation, cite the figure, then give 1–2 sentences of interpretation. Pull exact numbers from `summary_table.csv`.

**Figures to embed:**

| Figure | Placement | Caption |
|--------|-----------|---------|
| `fig1_learning_curves_2x2.png` | Full-width, top of §5 | Success rate vs. environment steps across all 4 environments. Shaded region = mean ± std over 3 seeds. |
| `sample_efficiency_50pct.png` | Left half-column | Steps to 50% success rate. — = did not reach threshold within budget. |
| `sample_efficiency_80pct.png` | Right half-column | Steps to 80% success rate. |
| `wallclock.png` | Half-column | Wall-clock minutes to reach 80% success. |
| `generalization_heatmap.png` | Half-column | Zero-shot success rate on 16×16 grid variants. |

**Result statements to fill in with real numbers:**

- R1: On Empty-8×8, all three agents converge. PPO reaches 80% in [X] steps; Dyna-Q in [Y]; Dreamer in [Z].
- R2: On DoorKey-8×8, model-based methods show clear advantage. Dreamer reaches 50% success [A]× faster than PPO.
- R3: On MultiRoom and KeyCorridor, PPO fails to reach 80% within budget; Dreamer achieves [B]% final success rate.
- R4: Wall-clock cost: Dreamer requires [C]× more wall-clock minutes than PPO to reach 80% on DoorKey.
- R5: Zero-shot generalization to 16×16: Dreamer achieves [D]% vs PPO [E]%.

---

### §6 Discussion & Failure Analysis (~1.0 page)

**When does planning help?**
Model-based advantage grows with task horizon. On Empty-8×8 (short horizon), PPO is competitive. On MultiRoom and KeyCorridor (long horizon), world-model planning enables credit assignment that random exploration cannot achieve within the step budget.

**Failure modes (one paragraph per agent):**
- *PPO*: Random exploration failure in long-horizon tasks. Without a model, the agent must stumble upon the reward by chance — exponentially unlikely in sparse-reward environments with many sequential subgoals.
- *Dyna-Q*: Model compounding error. The neural transition model accumulates prediction errors over multi-step rollouts, leading to hallucinated transitions that mislead the Q-network. Worst on KeyCorridor where dynamics involve key-door interactions.
- *Dreamer*: Cold-start latency. The RSSM requires sufficient real experience before the world model is reliable enough to support imagined training. This makes Dreamer slower to show early progress despite higher asymptotic sample efficiency.

**Practical framework (decision table):**

| Scenario | Recommended Agent | Reason |
|----------|------------------|--------|
| Short horizon, wall-clock budget tight | PPO | Lowest overhead |
| Long horizon, interaction cost high | Dreamer | Best sample efficiency |
| Middle ground, model interpretability needed | Dyna-Q | Explicit transition model |
| Zero-shot generalization required | Dreamer | World model encodes structure |

---

### §7 Conclusion & Author Contributions (~0.5 page)

**Conclusion (3 sentences):**
1. We benchmarked PPO, Dyna-Q, and Dreamer on four MiniGrid sparse-reward environments, measuring sample efficiency, wall-clock cost, and zero-shot generalization.
2. Model-based methods — especially Dreamer — consistently outperform PPO in sample efficiency on long-horizon tasks, at the cost of higher per-step compute.
3. These results suggest a clear selection criterion: prefer world-model-based RL when environment interactions are expensive and tasks require multi-step credit assignment.

**Author Contributions (equal split):**
- *Pranav Manimaran*: Agent implementation (PPO, Neural Dyna-Q, Dreamer), training infrastructure (GPU-parallel Kaggle notebook), experimental execution, result visualization.
- *Vandana Mansur*: Experimental design, hyperparameter study, quantitative analysis, report writing and editing.

---

### References (BibTeX keys to include)

```bibtex
@article{schulman2017ppo,
  title={Proximal Policy Optimization Algorithms},
  author={Schulman, John and Wolski, Filip and ...},
  year={2017}
}
@article{sutton1990dyna,
  title={Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming},
  author={Sutton, Richard S},
  year={1990}
}
@article{hafner2020dreamerv2,
  title={Mastering Atari with Discrete World Models},
  author={Hafner, Danijar and ...},
  year={2020}
}
@article{chevalier2023minigrid,
  title={Minigrid \& Miniworld: Modular \& Customizable Reinforcement Learning Environments},
  author={Chevalier-Boisvert, Maxime and ...},
  year={2023}
}
@article{mnih2015dqn,
  title={Human-level control through deep reinforcement learning},
  author={Mnih, Volodymyr and ...},
  year={2015}
}
```

---

## Verification Checklist

- [ ] `pdflatex main.tex` builds without errors
- [ ] `pdfinfo main.pdf | grep Pages` returns ≤ 8
- [ ] All 5 figures render (no missing files)
- [ ] Summary table numbers match `summary_table.csv` exactly
- [ ] Author contributions section present
- [ ] References formatted correctly (at least 5 citations)
- [ ] Both author names spelled correctly on title page
