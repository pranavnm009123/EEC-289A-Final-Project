# Presentation Plan — EEC 289A Final Project

**Title:** Sample Efficiency in Sparse-Reward Navigation: Model-Free vs. Model-Based RL  
**Presenters:** Pranav Manimaran, Vandana Mansur  
**Course:** EEC 289A — Sensorimotor Learning, UC Davis, Spring 2026  
**Duration:** 10 minutes total (+ Q&A)  
**Format:** Slide deck (PowerPoint / Google Slides / Beamer)

---

## Overview

| Metric | Value |
|--------|-------|
| Total slides | 12 |
| Time per slide (avg) | ~50 seconds |
| Pranav presents | Slides 1, 2, 5, 6, 7, 8, 9 (≈5 min) |
| Vandana presents | Slides 3, 4, 10, 11, 12 (≈5 min) |
| Shared | Slide 12 (closing) |

---

## Presenter Assignment Summary

| Slides | Presenter | Approx Time |
|--------|-----------|-------------|
| 1 — Title | Both (stand together) | 20 sec |
| 2 — Agenda | Pranav | 30 sec |
| 3 — Motivation | Vandana | 60 sec |
| 4 — The Core Question | Vandana | 60 sec |
| 5 — PPO | Pranav | 45 sec |
| 6 — Dyna-Q | Pranav | 45 sec |
| 7 — Dreamer | Pranav | 60 sec |
| 8 — Experimental Setup | Pranav | 45 sec |
| 9 — Learning Curves | Pranav | 60 sec |
| 10 — Sample Efficiency | Vandana | 60 sec |
| 11 — Wall-Clock + Generalization | Vandana | 60 sec |
| 12 — Discussion + Takeaways | Vandana | 60 sec |
| 13 — Thank You / Q&A | Both | 20 sec |

---

## Slide-by-Slide Detail

---

### Slide 1 — Title
**Presenter:** Both  
**Time:** 20 seconds  

**Content:**
- Title: "Sample Efficiency in Sparse-Reward Navigation: Model-Free vs. Model-Based RL"
- Authors: Pranav Manimaran, Vandana Mansur
- Course: EEC 289A — Sensorimotor Learning, UC Davis, Spring 2026

**Visual:** Clean title slide, no figures. Use a subtle background (dark grid-world pattern or plain academic style).

**Speaker notes (Pranav):** "Hi everyone — I'm Pranav, and this is Vandana. Today we're going to talk about a core challenge in reinforcement learning: how efficiently can agents learn in environments where reward is extremely sparse?"

---

### Slide 2 — Agenda
**Presenter:** Pranav  
**Time:** 30 seconds  

**Content (5 bullets):**
1. Motivation — why sample efficiency matters
2. Three algorithms — PPO, Dyna-Q, Dreamer
3. Experimental setup — 4 MiniGrid environments
4. Results — learning curves, sample efficiency, generalization
5. Takeaways — when to use which method

**Visual:** Numbered list, no figures. Optionally highlight current item as presentation progresses.

**Speaker notes:** "Here's what we'll cover. We'll keep methods brief and spend most of our time on results."

---

### Slide 3 — Motivation
**Presenter:** Vandana  
**Time:** 60 seconds  

**Content:**
- **Headline:** "Every environment interaction has a real cost"
- 3 bullet points:
  - Robotics: physical wear, time, safety risks
  - Healthcare: patient burden, ethical constraints
  - Autonomous navigation: expensive simulation or real-world trials
- **The hard case:** Sparse reward — agent gets +1 only when it reaches the goal, 0 everywhere else. Must discover a long sequence of correct actions before receiving any learning signal.
- **The question:** Can model-based methods dramatically reduce the number of interactions needed?

**Visual:** Two-panel graphic:
- Left: cartoon robot failing repeatedly (dense reward label: "feedback every step")
- Right: sparse reward timeline showing long stretches of 0 reward then a single +1 at the end
- OR: just a reward-vs-time plot showing dense vs sparse signal

**Speaker notes (Vandana):** "In the real world, you often can't afford millions of environment interactions. Sparse reward makes this dramatically worse — the agent gets almost no feedback until it actually solves the task. That's the setting we're studying."

---

### Slide 4 — The Core Question
**Presenter:** Vandana  
**Time:** 60 seconds  

**Content:**
- **Headline:** "Model-free vs. model-based RL — does planning help?"
- Left column: **Model-free** (PPO) — learns policy directly from experience, no environment model
- Right column: **Model-based** (Dyna-Q, Dreamer) — also learns a predictive model of the environment; uses it to generate synthetic experience and plan ahead
- **Three research questions:**
  1. Which agent reaches competent performance with the fewest real environment steps?
  2. How much wall-clock time does planning overhead add?
  3. Do world models generalize zero-shot to larger environments?

**Visual:** Simple split diagram:
```
Real Experience
      |
  ┌───┴────┐
Model-Free  Model-Based
(PPO)      (Dyna-Q, Dreamer)
  ↓              ↓
Policy     Model + Policy
           (plan ahead)
```

**Speaker notes:** "These are the three questions our experiments answer. The tension is: model-based methods might learn faster, but they add compute overhead. We want to know exactly when that tradeoff is worth it."

---

### Slide 5 — Agent 1: PPO (Model-Free Baseline)
**Presenter:** Pranav  
**Time:** 45 seconds  

**Content:**
- **Proximal Policy Optimization** — Schulman et al. 2017
- Shared CNN encoder → policy head + value head
- Clipped surrogate objective: prevents large destructive policy updates
- GAE advantage estimation (λ = 0.95)
- Key hyperparams: lr = 3×10⁻⁴, clip ε = 0.2, 500K steps
- **No environment model** — learns purely from rollout experience

**Visual:** Simple architecture flow:
`Observation → CNN (256) → Linear → [Policy Head | Value Head]`

**Speaker notes:** "PPO is our model-free baseline — well-established, stable, and widely used. It has no concept of how the world works; it just optimizes its policy from experience."

---

### Slide 6 — Agent 2: Neural Dyna-Q (Classical Model-Based)
**Presenter:** Pranav  
**Time:** 45 seconds  

**Content:**
- **Dyna-Q** — Sutton 1990, with neural networks
- DQN (online + target network) + neural transition model: (s, a) → (s′, r)
- **Planning ratio k = 5**: for every 1 real step, 5 extra Q-updates using model-generated transitions
- Real experience trains both the Q-network and the transition model
- Key hyperparams: Q-net lr = 1×10⁻⁴, model lr = 1×10⁻³, 500K steps

**Visual:** Dyna loop diagram:
```
Environment → real (s,a,r,s') → Train Q-net
                               ↓
                         Train Model
                               ↓
              Model generates 5× synthetic transitions
                               ↓
                     Q-net updates (×5)
```

**Speaker notes:** "Dyna-Q augments standard Q-learning with a learned model. After each real step, it simulates 5 extra transitions — effectively 5× data efficiency from the same real experience."

---

### Slide 7 — Agent 3: Dreamer (World Model)
**Presenter:** Pranav  
**Time:** 60 seconds  

**Content:**
- **DreamerV2** — Hafner et al. 2020
- Learns a compact latent world model using RSSM (Recurrent State Space Model)
  - CNN encodes observations → discrete latent z (32×32 categories)
  - GRU maintains recurrent hidden state h (dim 512)
  - World model predicts: next latent, reward, episode termination
- **Actor-critic trained entirely in imagination** — never touches real environment during policy learning
- Imagination horizon H = 15 steps
- Key hyperparams: world model lr = 6×10⁻⁴, actor/critic lr = 8×10⁻⁵, only 200K real steps

**Visual:** RSSM diagram:
```
Observation → CNN Encoder → z (latent)
                               ↓
              h (GRU) ←──── h + z (RSSM state)
                               ↓
              Imagine H=15 steps in latent space
                               ↓
              Train Actor-Critic on imagined rollouts
```

**Speaker notes:** "Dreamer is the most sophisticated of the three. It learns an entire world model in latent space and trains its policy purely through imagination — so it needs far fewer real interactions."

---

### Slide 8 — Experimental Setup
**Presenter:** Pranav  
**Time:** 45 seconds  

**Content:**
- **4 MiniGrid environments** (increasing horizon / difficulty):

| Environment | Challenge |
|-------------|-----------|
| Empty-8×8 | Open room, short horizon (sanity check) |
| DoorKey-8×8 | Pick up key → unlock door → reach goal |
| MultiRoom-N4-S5 | Traverse 4 connected rooms |
| KeyCorridor-S3-R1 | Multiple locked doors, longest horizon |

- All: sparse reward (+1 at goal only)
- 3 random seeds (42, 123, 777) per (agent, env) pair → **36 total runs**
- Evaluated every 2,048 steps on 100 episodes
- After training: zero-shot eval on 16×16 variants (no fine-tuning)

**Visual:** 2×2 grid of MiniGrid screenshots (one per environment), labeled.

**Speaker notes:** "We deliberately chose environments of increasing complexity. Empty-8×8 is a sanity check. KeyCorridor is where we expect model-free methods to struggle most."

---

### Slide 9 — Results: Learning Curves
**Presenter:** Pranav  
**Time:** 60 seconds  

**Content:**
- **Main figure:** `fig1_learning_curves_2x2.png` — full slide (or near full)
- Caption: Success rate vs. environment steps, mean ± std across 3 seeds
- Talk through the 4 panels:
  - Empty: all agents converge ✓
  - DoorKey: Dreamer pulls ahead first
  - MultiRoom: PPO flat-lines; Dyna-Q struggles; Dreamer climbs
  - KeyCorridor: only model-based methods make meaningful progress

**Key observation to state:** "The advantage of model-based methods grows with task horizon. On the simplest task, all three work. On the hardest tasks, PPO barely learns at all."

**Visual:** `fig1_learning_curves_2x2.png` (full-width, 2×2 panel grid)

**Speaker notes:** "This is the headline result. Look at the bottom row — MultiRoom and KeyCorridor. PPO is essentially flat. Dreamer continues climbing because it can plan across longer horizons through imagination."

---

### Slide 10 — Results: Sample Efficiency
**Presenter:** Vandana  
**Time:** 60 seconds  

**Content:**
- Two bar charts side by side: steps to 50% success | steps to 80% success
- Pull **3 specific numbers** from `summary_table.csv` to state explicitly:
  - Example: "Dreamer reaches 80% success on DoorKey in [X]K steps vs PPO's [Y]K — [Z]× more efficient"
  - Example: "PPO never reached 80% on MultiRoom within 500K steps"
  - Example: "On Empty-8×8, PPO is fastest — [A]K steps vs Dreamer's [B]K"
- Key message: model-based wins on hard tasks; PPO competitive on easy tasks

**Visual:** `sample_efficiency_50pct.png` and `sample_efficiency_80pct.png` side by side.  
Use a callout arrow or annotation on the chart pointing to the most dramatic difference.

**Speaker notes (Vandana):** "Here are the numbers. These bar charts show how many steps each agent needs to hit our performance thresholds. The missing bars mean the agent never reached that threshold within the full step budget."

---

### Slide 11 — Wall-Clock Cost + Generalization
**Presenter:** Vandana  
**Time:** 60 seconds  

**Two-panel slide — left: wall-clock, right: generalization**

**Left panel — Wall-Clock:**
- Figure: `wallclock.png`
- Key message: Dreamer takes [X]× more wall-clock time than PPO to reach 80% on DoorKey
- Interpretation: planning overhead is real — model-based is NOT free
- "The sample efficiency gain has a compute cost. The question is whether your environment interaction cost exceeds your compute cost."

**Right panel — Generalization:**
- Figure: `generalization_heatmap.png`
- Zero-shot success on 16×16 grids (agents trained on 8×8 only)
- Key message: Dreamer generalizes best — world model encodes reusable structure
- Numbers from `generalization_results.csv`

**Speaker notes:** "Two things here. Left: model-based isn't free — Dreamer is slower per training run. Right: but it generalizes better zero-shot, because the world model has learned how grid environments actually work, not just where the goal is in this specific map."

---

### Slide 12 — Discussion + Key Takeaways
**Presenter:** Vandana  
**Time:** 60 seconds  

**Content:**

**Failure modes (quick table — 3 rows):**
| Agent | Failure mode |
|-------|-------------|
| PPO | Random exploration collapses on long-horizon tasks |
| Dyna-Q | Model error compounds over multi-step rollouts |
| Dreamer | Cold-start latency — slow early progress |

**3 Takeaways (matching the 3 research questions from Slide 4):**
1. **Sample efficiency:** Dreamer is [X]× more efficient than PPO on long-horizon tasks; Dyna-Q occupies the middle ground.
2. **Wall-clock cost:** Model-based methods add [Y]× training overhead — worth it when real interaction is expensive.
3. **Generalization:** World models transfer structure to larger grids; PPO does not.

**Practical recommendation:**
- Short horizon / compute-limited → use PPO
- Long horizon / interaction-limited → use Dreamer
- Need interpretable model → use Dyna-Q

**Visual:** Failure-mode table + 3 numbered takeaways in large text.

**Speaker notes:** "To summarize: model-based RL pays off, but not everywhere. The longer the task horizon and the more expensive each real interaction, the more you want Dreamer. If you just need something simple that works, PPO is hard to beat."

---

### Slide 13 — Thank You / Q&A
**Presenter:** Both  
**Time:** 20 seconds  

**Content:**
- "Thank you!"
- "Questions?"
- **Author contributions (equal split):**
  - Pranav Manimaran: Agent implementation, training infrastructure, experimental execution, visualization
  - Vandana Mansur: Experimental design, quantitative analysis, report writing and editing
- Optional: link to code / results notebook

**Visual:** Clean closing slide — title, author names, "Questions?" in large text.

---

## Design Recommendations for the Slide Deck

**Style:**
- Use a clean academic template (dark background or white — avoid busy themes)
- Consistent font: title 28–32pt, body 18–22pt, captions 14pt
- Limit to 4–5 bullet points per slide maximum
- One big idea per slide

**Figures:**
- All figures come from `results/extracted/plots/` (replace with final results plots when zip arrives)
- Use the exact same figures as the report for consistency
- Crop whitespace from PNGs before inserting

**Color palette:**
- PPO → blue
- Dyna-Q → orange
- Dreamer → green
(Match whatever colors the `visualize.py` script used in the learning curve plots)

**Transitions:**
- None or simple fade — keep it professional
- No animations on individual bullet points (wastes time in a 10-min talk)

---

## 10-Minute Pacing Guide

| Time mark | Where you should be |
|-----------|---------------------|
| 0:00 | Slide 1 — Title |
| 0:20 | Slide 2 — Agenda |
| 0:50 | Slide 3 — Motivation |
| 1:50 | Slide 4 — Core Question |
| 2:50 | Slide 5 — PPO |
| 3:35 | Slide 6 — Dyna-Q |
| 4:20 | Slide 7 — Dreamer |
| 5:20 | Slide 8 — Setup |
| 6:05 | Slide 9 — Learning Curves |
| 7:05 | Slide 10 — Sample Efficiency |
| 8:05 | Slide 11 — Wall-Clock + Generalization |
| 9:05 | Slide 12 — Discussion + Takeaways |
| 10:00 | Slide 13 — Q&A |

---

## What to Fill In When Results Are Ready

All `[X]`, `[Y]`, `[Z]`, `[A]`, `[B]` placeholders in slides 9–12 need to be replaced with real numbers from:
- `results/extracted/tables/summary_table.csv` — steps to threshold, final success rate
- `results/extracted/tables/generalization_results.csv` — 16×16 zero-shot rates
- `results/extracted/plots/` — insert final PNG figures
