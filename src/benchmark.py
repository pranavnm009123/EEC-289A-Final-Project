"""
EEC 289A — Sensorimotor Learning
Benchmarking script: PPO vs Neural Dyna-Q vs Dreamer-style World Model
on MiniGrid sparse-reward navigation tasks.

Team: Pranav Manimaran, Vandana Mansur

Usage:
    python benchmark.py --agent ppo        --env doorkey --seed 42 --total_steps 500000
    python benchmark.py --agent dynaq      --env doorkey --seed 42 --total_steps 500000
    python benchmark.py --agent dreamer    --env doorkey --seed 42 --total_steps 200000
"""

import argparse
import collections
import csv
import os
import random
import re
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

ENV_MAP = {
    "empty":       "MiniGrid-Empty-8x8-v0",
    "doorkey":     "MiniGrid-DoorKey-8x8-v0",
    "multiroom":   "MiniGrid-MultiRoom-N4-S5-v0",
    "keycorridor": "MiniGrid-KeyCorridorS3R1-v0",
}

MAX_STEPS_MAP = {
    "empty":       128,
    "doorkey":     300,
    "multiroom":   250,
    "keycorridor": 200,
}

# 16x16 variants for zero-shot generalization evaluation (plan.txt §5.3)
GEN_ENV_MAP = {
    "empty":       "MiniGrid-Empty-16x16-v0",
    "doorkey":     "MiniGrid-DoorKey-16x16-v0",
    "multiroom":   "MiniGrid-MultiRoom-N6-v0",
    "keycorridor": "MiniGrid-KeyCorridorS4R3-v0",
}
GEN_MAX_STEPS_MAP = {
    "empty": 512, "doorkey": 1200, "multiroom": 1000, "keycorridor": 800,
}


class NormalizeObs(gym.ObservationWrapper):
    """Transpose (H,W,C) -> (C,H,W) and normalize to [0,1] float32."""

    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape  # (H, W, C)
        H, W, C = obs_shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(C, H, W), dtype=np.float32
        )
        # Per-channel max values from MiniGrid encoding
        self._channel_max = np.array([10.0, 5.0, 2.0], dtype=np.float32)

    def observation(self, obs):
        obs = obs.astype(np.float32)                    # (H, W, C)
        obs = obs / self._channel_max[None, None, :]    # normalize per channel
        obs = obs.transpose(2, 0, 1)                    # (C, H, W)
        return obs.copy()


class MaskDoneAction(gym.ActionWrapper):
    """Drop action index 6 (MiniGrid 'done' no-op) to keep the effective
    action space at 6.  All indices 0-5 pass through unchanged."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(6)

    def action(self, act):
        return int(act)   # 0-5 map 1:1; index 6 is never sampled


def make_env(env_key: str, seed: int, max_steps: int = None):
    env_id = ENV_MAP[env_key]
    if max_steps is None:
        max_steps = MAX_STEPS_MAP[env_key]
    env = gym.make(env_id, max_steps=max_steps)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    env = NormalizeObs(env)
    env = MaskDoneAction(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


# ---------------------------------------------------------------------------
# Shared CNN Encoder
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """
    Encodes a (B, C, H, W) MiniGrid observation into a (B, 256) feature vector.
    Works for any grid size thanks to adaptive pooling.
    """

    def __init__(self, obs_shape):
        super().__init__()
        C, H, W = obs_shape
        # GroupNorm instead of BatchNorm: works correctly with batch_size=1
        # during single-step rollout collection (BatchNorm is undefined at B=1).
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(inplace=True),
        )
        # Adaptive pool to fixed 4x4 so we work with any grid resolution
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class CSVLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = None
        self._start = time.time()

    def write(self, row: dict):
        full_row = {**row, "wall_time_sec": round(time.time() - self._start, 2)}
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(full_row.keys()))
            self._writer.writeheader()
        self._writer.writerow(full_row)
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.close()

    def __del__(self):
        if hasattr(self, '_file'):
            self.close()


def evaluate_agent(act_fn, env_key: str, seed: int, n_episodes: int = 20,
                   device: torch.device = torch.device("cpu")) -> dict:
    """Run n_episodes deterministic episodes and return metrics dict."""
    env = make_env(env_key, seed=seed + 10000)
    successes, ep_lengths, ep_returns = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_len = 0
        ep_ret = 0.0
        while not done:
            with torch.no_grad():
                action = act_fn(obs_to_tensor(obs, device))
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_len += 1
            ep_ret += float(reward)
        successes.append(1 if (terminated and ep_ret > 0) else 0)
        ep_lengths.append(ep_len)
        ep_returns.append(ep_ret)
    env.close()
    return {
        "success_rate": float(np.mean(successes)),
        "mean_ep_length": float(np.mean(ep_lengths)),
        "mean_return": float(np.mean(ep_returns)),
    }


def save_checkpoint(state_dict: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)


def load_checkpoint_and_eval_generalization(
        agent: str, ckpt_path: str, env_key: str,
        obs_shape, n_actions: int, device: torch.device,
        n_episodes: int = 50) -> float:
    """Zero-shot eval of a trained checkpoint on the 16x16 variant of env_key."""
    if env_key not in GEN_ENV_MAP:
        print(f"  [gen_eval] No 16x16 variant defined for '{env_key}' — skipping.")
        return float("nan")

    gen_env_id = GEN_ENV_MAP[env_key]
    max_s = GEN_MAX_STEPS_MAP[env_key]
    env = gym.make(gen_env_id, max_steps=max_s)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    env = NormalizeObs(env)
    env = MaskDoneAction(env)
    env.reset(seed=99999)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if agent == "ppo":
        net = PPOActorCritic(obs_shape, n_actions).to(device)
        net.load_state_dict(ckpt["net"])
        net.eval()
        act_fn = lambda o: net.act(o, deterministic=True)[0]
        dreamer_mode = False

    elif agent == "dynaq":
        online_net = DynaQNetwork(obs_shape, n_actions).to(device)
        online_net.load_state_dict(ckpt["online_net"])
        online_net.eval()
        act_fn = lambda o: online_net(o).argmax(dim=1).item()
        dreamer_mode = False

    elif agent == "dreamer":
        z_dim = 32
        rssm_g = RSSM(obs_shape, n_actions).to(device)
        ac_g   = LatentActorCritic(state_dim=512 + z_dim, n_actions=n_actions).to(device)
        rssm_g.load_state_dict(ckpt["rssm"])
        ac_g.actor.load_state_dict(ckpt["actor"])
        rssm_g.eval(); ac_g.eval()
        act_fn = None
        dreamer_mode = True
    else:
        env.close()
        return float("nan")

    successes = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False; ep_ret = 0.0; terminated = False

        if dreamer_mode:
            h = rssm_g.init_hidden(1, device)
            z = torch.zeros(1, z_dim, device=device)
            prev_a = torch.zeros(1, dtype=torch.long, device=device)

        while not done:
            with torch.no_grad():
                obs_t = obs_to_tensor(obs, device)
                if dreamer_mode:
                    h, z_post, *_ = rssm_g.observe_step(h, obs_t, prev_a, z)
                    z = z_post
                    state = rssm_g.world_state(h, z)
                    action_t, *_ = ac_g.act(state, deterministic=True)
                    action = action_t.item()
                    prev_a = action_t.unsqueeze(0) if action_t.dim() == 0 else action_t
                else:
                    action = act_fn(obs_t)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
        successes.append(1 if (terminated and ep_ret > 0) else 0)

    env.close()
    return float(np.mean(successes))


def _append_gen_result(path: str, agent: str, env: str, seed: int, rate: float):
    """Append a row to the generalization results CSV (creates file + header if new)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["agent", "env", "seed", "gen_success_rate_16x16"])
        if write_header:
            writer.writeheader()
        writer.writerow({"agent": agent, "env": env, "seed": seed,
                         "gen_success_rate_16x16": round(rate, 4)})


def _find_latest_checkpoint(save_dir: str, agent: str, env_key: str, seed: int):
    """Return path to the latest checkpoint file for this (agent, env, seed)."""
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    prefix = f"{agent}_{env_key}_seed{seed}_step"
    candidates = [
        f for f in os.listdir(ckpt_dir)
        if f.startswith(prefix) and f.endswith(".pt")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda f: int(re.search(r"_step(\d+)\.pt$", f).group(1)))
    return os.path.join(ckpt_dir, candidates[-1])


# ===========================================================================
# AGENT 1: PPO
# ===========================================================================

class PPOActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions: int):
        super().__init__()
        self.encoder = CNNEncoder(obs_shape)
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs: torch.Tensor):
        z = self.encoder(obs)
        dist = Categorical(logits=self.actor(z))
        value = self.critic(z).squeeze(-1)
        return dist, value

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist, value = self(obs)
        action = dist.mode if deterministic else dist.sample()
        return action.item(), dist.log_prob(action), value


class PPORolloutBuffer:
    def __init__(self, n_steps: int, obs_shape, device: torch.device):
        self.n_steps = n_steps
        self.device = device
        self.obs        = torch.zeros(n_steps, *obs_shape, device=device)
        self.actions    = torch.zeros(n_steps, dtype=torch.long, device=device)
        self.log_probs  = torch.zeros(n_steps, device=device)
        self.rewards    = torch.zeros(n_steps, device=device)
        self.values     = torch.zeros(n_steps, device=device)
        self.dones      = torch.zeros(n_steps, device=device)
        self.advantages = torch.zeros(n_steps, device=device)
        self.returns    = torch.zeros(n_steps, device=device)
        self.ptr = 0

    def store(self, obs, action, log_prob, reward, value, done):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr]   = reward
        self.values[self.ptr]    = value
        self.dones[self.ptr]     = done
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float = 0.99,
                    gae_lambda: float = 0.95):
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_value = last_value if t == self.n_steps - 1 else float(self.values[t + 1])
            next_non_terminal = 1.0 - float(self.dones[t])
            delta = (float(self.rewards[t])
                     + gamma * next_value * next_non_terminal
                     - float(self.values[t]))
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get_minibatches(self, batch_size: int):
        indices = torch.randperm(self.n_steps, device=self.device)
        for start in range(0, self.n_steps, batch_size):
            idx = indices[start:start + batch_size]
            yield (self.obs[idx], self.actions[idx], self.advantages[idx],
                   self.returns[idx], self.log_probs[idx])

    def reset(self):
        self.ptr = 0


def train_ppo(env_key: str, seed: int, total_steps: int, save_dir: str,
              device: torch.device):
    set_seed(seed)
    env = make_env(env_key, seed)
    obs_shape = env.observation_space.shape   # (C, H, W)
    n_actions = env.action_space.n

    net = PPOActorCritic(obs_shape, n_actions).to(device)
    initial_lr = 3e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr, eps=1e-5)

    # Hyperparameters
    n_steps       = 2048
    n_epochs      = 10
    batch_size    = 64
    gamma         = 0.99
    gae_lambda    = 0.95
    clip_eps      = 0.2
    value_coef    = 0.5
    entropy_coef  = 0.01
    max_grad_norm = 0.5
    eval_interval = 10_000
    ckpt_interval = 50_000

    buffer = PPORolloutBuffer(n_steps, obs_shape, device)
    log_path = os.path.join(save_dir, "logs", f"ppo_{env_key}_seed{seed}.csv")
    logger = CSVLogger(log_path)

    global_step = 0
    last_eval_step = -eval_interval
    last_ckpt_step = -ckpt_interval
    obs, _ = env.reset()

    # Linear LR decay schedule
    def lr_at_step(s):
        frac = 1.0 - s / total_steps
        for pg in optimizer.param_groups:
            pg["lr"] = initial_lr * frac

    print(f"[PPO] env={env_key} seed={seed} total_steps={total_steps}")

    while global_step < total_steps:
        buffer.reset()
        net.train()

        # ---- Rollout collection ----
        for _ in range(n_steps):
            obs_t = obs_to_tensor(obs, device)
            with torch.no_grad():
                action, log_prob, value = net.act(obs_t)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.store(
                obs_t.squeeze(0), action, log_prob.item(),
                reward, value.item(), float(done)
            )
            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs
            global_step += 1

        # Bootstrap last value
        with torch.no_grad():
            _, last_val = net(obs_to_tensor(obs, device))
        buffer.compute_gae(last_val.item(), gamma, gae_lambda)

        # ---- PPO update ----
        p_losses, v_losses, entropies, grad_norms = [], [], [], []
        for _ in range(n_epochs):
            for obs_b, act_b, adv_b, ret_b, old_lp_b in buffer.get_minibatches(batch_size):
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
                dist, value = net(obs_b)
                new_lp = dist.log_prob(act_b)
                entropy = dist.entropy().mean()

                ratio = (new_lp - old_lp_b).exp()
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, ret_b)
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm).item()
                optimizer.step()

                p_losses.append(policy_loss.item())
                v_losses.append(value_loss.item())
                entropies.append(entropy.item())
                grad_norms.append(gn)

        lr_at_step(global_step)

        # ---- Eval + log ----
        if global_step - last_eval_step >= eval_interval:
            last_eval_step = global_step
            net.eval()
            act_fn = lambda o: net.act(o, deterministic=True)[0]
            metrics = evaluate_agent(act_fn, env_key, seed, n_episodes=20,
                                     device=device)
            row = {
                "step": global_step,
                "success_rate": metrics["success_rate"],
                "mean_ep_length": metrics["mean_ep_length"],
                "mean_return": metrics["mean_return"],
                "policy_loss": np.mean(p_losses),
                "value_loss": np.mean(v_losses),
                "entropy": np.mean(entropies),
                "grad_norm": np.mean(grad_norms),
            }
            logger.write(row)
            print(f"  step={global_step:>7d}  success={metrics['success_rate']:.3f}"
                  f"  p_loss={np.mean(p_losses):.4f}  v_loss={np.mean(v_losses):.4f}")
            net.train()

        if global_step - last_ckpt_step >= ckpt_interval:
            last_ckpt_step = global_step
            ckpt_path = os.path.join(save_dir, "checkpoints",
                                     f"ppo_{env_key}_seed{seed}_step{global_step}.pt")
            save_checkpoint({"net": net.state_dict(), "step": global_step}, ckpt_path)

    logger.close()
    env.close()
    print(f"[PPO] Training complete. Logs: {log_path}")


# ===========================================================================
# AGENT 2: NEURAL DYNA-Q
# ===========================================================================

class DynaQNetwork(nn.Module):
    """CNN encoder + Q-value head. Separate from encoder to allow joint training."""

    def __init__(self, obs_shape, n_actions: int):
        super().__init__()
        self.encoder = CNNEncoder(obs_shape)
        self.q_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_actions),
        )
        nn.init.orthogonal_(self.q_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.q_head[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.encoder(obs))

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)


class NeuralTransitionModel(nn.Module):
    """
    Parametric transition model: (z, a) -> (z_next, r, done_logit)
    z is the 256-dim CNN latent. Action is embedded before concatenation.
    Separate heads for next-z, reward, and episode-termination so the
    planning loop can apply a proper (1-done) mask on imagined Q-targets.
    """

    def __init__(self, latent_dim: int = 256, n_actions: int = 6):
        super().__init__()
        self.action_embed = nn.Embedding(n_actions, 32)
        self.body = nn.Sequential(
            nn.Linear(latent_dim + 32, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
        )
        self.next_z_head  = nn.Linear(512, latent_dim)
        self.reward_head  = nn.Linear(512, 1)
        self.done_head    = nn.Linear(512, 1)   # logit; sigmoid → P(episode ends)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, actions: torch.Tensor):
        a_emb = self.action_embed(actions)             # (B, 32)
        x = torch.cat([z, a_emb], dim=-1)              # (B, 288)
        feat = self.body(x)                             # (B, 512)
        next_z     = self.next_z_head(feat)             # (B, latent_dim)
        reward     = self.reward_head(feat).squeeze(-1) # (B,)
        done_logit = self.done_head(feat).squeeze(-1)   # (B,)
        return next_z, reward, done_logit


class ReplayBuffer:
    """Circular replay buffer storing (obs, action, reward, next_obs, done)."""

    def __init__(self, capacity: int, obs_shape, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.obs      = torch.zeros(capacity, *obs_shape)
        self.actions  = torch.zeros(capacity, dtype=torch.long)
        self.rewards  = torch.zeros(capacity)
        self.next_obs = torch.zeros(capacity, *obs_shape)
        self.dones    = torch.zeros(capacity)
        self.ptr = 0
        self.size = 0

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = torch.as_tensor(obs)
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = torch.as_tensor(next_obs)
        self.dones[self.ptr]    = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device = None):
        idx = np.random.randint(0, self.size, size=batch_size)
        d = device if device is not None else self.device
        return (
            self.obs[idx].to(d),
            self.actions[idx].to(d),
            self.rewards[idx].to(d),
            self.next_obs[idx].to(d),
            self.dones[idx].to(d),
        )

    def __len__(self):
        return self.size


def train_dynaq(env_key: str, seed: int, total_steps: int, save_dir: str,
                device: torch.device, planning_k: int = 5):
    set_seed(seed)
    env = make_env(env_key, seed)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    online_net = DynaQNetwork(obs_shape, n_actions).to(device)
    target_net = deepcopy(online_net).to(device)
    target_net.eval()
    for p in target_net.parameters():
        p.requires_grad_(False)

    transition_model = NeuralTransitionModel(latent_dim=256, n_actions=n_actions).to(device)

    q_optimizer     = torch.optim.Adam(online_net.parameters(), lr=1e-4, eps=1e-5)
    model_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-3, eps=1e-5)

    buffer = ReplayBuffer(capacity=50_000, obs_shape=obs_shape, device=torch.device("cpu"))

    # Hyperparameters
    gamma           = 0.99
    eps_start       = 1.0
    eps_end         = 0.05
    eps_decay_steps = 50_000
    target_update   = 500
    batch_size      = 64
    warmup_steps    = 1_000
    eval_interval   = 10_000
    ckpt_interval   = 50_000

    log_suffix = f"_k{planning_k}" if planning_k != 5 else ""
    log_path = os.path.join(save_dir, "logs", f"dynaq_{env_key}_seed{seed}{log_suffix}.csv")
    logger = CSVLogger(log_path)

    obs, _ = env.reset()
    global_step = 0
    last_eval_step = -eval_interval
    last_ckpt_step = -ckpt_interval
    q_loss_ema = None
    model_loss_ema = None
    grad_norm_ema = None

    print(f"[Neural Dyna-Q] env={env_key} seed={seed} total_steps={total_steps} k={planning_k}")

    while global_step < total_steps:
        # ---- Epsilon-greedy action ----
        eps = max(eps_end, eps_start - (eps_start - eps_end) * global_step / eps_decay_steps)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = obs_to_tensor(obs, device)
                q_vals = online_net(obs_t)
                action = q_vals.argmax(dim=1).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, done)
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
        global_step += 1

        if len(buffer) < warmup_steps:
            continue

        # ---- Update Q-network on real transitions ----
        obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(batch_size, device=device)

        with torch.no_grad():
            # Double DQN: online net selects action, target net evaluates
            next_q_online = online_net(next_obs_b)
            best_next_act = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = target_net(next_obs_b).gather(1, best_next_act).squeeze(1)
            target_q = rew_b + gamma * next_q_target * (1.0 - done_b)

        current_q = online_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
        q_loss = F.smooth_l1_loss(current_q, target_q)

        q_optimizer.zero_grad()
        q_loss.backward()
        gn_q = nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0).item()
        q_optimizer.step()

        # ---- Update transition model on real transitions ----
        # Use detached latents so model training doesn't affect encoder
        with torch.no_grad():
            z_b      = online_net.encode(obs_b)
            next_z_b = online_net.encode(next_obs_b)

        pred_next_z, pred_reward, pred_done_logit = transition_model(z_b, act_b)
        model_loss = (F.mse_loss(pred_next_z, next_z_b)
                      + 0.5 * F.mse_loss(pred_reward, rew_b)
                      + 0.5 * F.binary_cross_entropy_with_logits(pred_done_logit, done_b))

        model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(transition_model.parameters(), max_norm=10.0)
        model_optimizer.step()

        grad_norm_ema = gn_q if grad_norm_ema is None else 0.99 * grad_norm_ema + 0.01 * gn_q

        # ---- Dyna planning: k imagined Q-updates ----
        for _ in range(planning_k):
            # Sample past real states from replay buffer
            p_obs_b, _, _, _, _ = buffer.sample(batch_size, device=device)
            with torch.no_grad():
                p_z_b = online_net.encode(p_obs_b)

            # Random imagined actions: standard Dyna sweeps all (s,a) pairs
            imag_actions = torch.randint(0, n_actions, (batch_size,), device=device)

            with torch.no_grad():
                imag_next_z, imag_reward, imag_done_logit = transition_model(p_z_b, imag_actions)
                imag_cont = 1.0 - torch.sigmoid(imag_done_logit)  # P(episode continues)

                # Q-target in latent space using target_net's q_head on imag_next_z
                # We pass imag_next_z directly into the Q head (bypass encoder)
                next_q_latent = target_net.q_head(imag_next_z)
                best_imag_act = online_net.q_head(imag_next_z).argmax(dim=1, keepdim=True)
                imag_next_q = next_q_latent.gather(1, best_imag_act).squeeze(1)
                imag_target_q = imag_reward + gamma * imag_next_q * imag_cont

            imag_current_q = online_net.q_head(p_z_b).gather(
                1, imag_actions.unsqueeze(1)).squeeze(1)
            imag_q_loss = F.smooth_l1_loss(imag_current_q, imag_target_q.detach())

            q_optimizer.zero_grad()
            imag_q_loss.backward()
            nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
            q_optimizer.step()

        # ---- Hard target network update ----
        if global_step % target_update == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Exponential moving average of losses for logging
        q_loss_ema     = q_loss.item()     if q_loss_ema     is None else 0.99 * q_loss_ema     + 0.01 * q_loss.item()
        model_loss_ema = model_loss.item() if model_loss_ema is None else 0.99 * model_loss_ema + 0.01 * model_loss.item()

        # ---- Eval + log ----
        if global_step - last_eval_step >= eval_interval:
            last_eval_step = global_step
            online_net.eval()
            metrics = evaluate_agent(
                lambda o: online_net(o).argmax(dim=1).item(),
                env_key, seed, n_episodes=20, device=device)
            row = {
                "step": global_step,
                "success_rate": metrics["success_rate"],
                "mean_ep_length": metrics["mean_ep_length"],
                "mean_return": metrics["mean_return"],
                "q_loss": q_loss_ema,
                "model_loss": model_loss_ema,
                "epsilon": round(eps, 4),
                "grad_norm": grad_norm_ema if grad_norm_ema is not None else 0.0,
            }
            logger.write(row)
            print(f"  step={global_step:>7d}  success={metrics['success_rate']:.3f}"
                  f"  q_loss={q_loss_ema:.4f}  m_loss={model_loss_ema:.4f}  eps={eps:.3f}")
            online_net.train()

        if global_step - last_ckpt_step >= ckpt_interval:
            last_ckpt_step = global_step
            ckpt_path = os.path.join(save_dir, "checkpoints",
                                     f"dynaq_{env_key}_seed{seed}_step{global_step}.pt")
            save_checkpoint({
                "online_net": online_net.state_dict(),
                "transition_model": transition_model.state_dict(),
                "step": global_step,
            }, ckpt_path)

    logger.close()
    env.close()
    print(f"[Neural Dyna-Q] Training complete. Logs: {log_path}")


# ===========================================================================
# AGENT 3: DREAMER-STYLE WORLD MODEL
# ===========================================================================

class RSSM(nn.Module):
    """
    Recurrent State Space Model.
    Deterministic path:  h_t = GRU(h_{t-1}, cat(z_{t-1}, a_embed_{t-1}))
    Stochastic path:     z_t ~ Normal(mu, sigma) from posterior q(z|h, o)
                              OR prior              p(z|h)
    World state = cat(h_t, z_t)   shape: (B, h_dim + z_dim)
    """

    def __init__(self, obs_shape, n_actions: int,
                 h_dim: int = 512, z_dim: int = 32):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_actions = n_actions

        # Obs encoder shared with world model
        self.obs_encoder = CNNEncoder(obs_shape)          # -> 256

        # Action embedding
        self.action_embed = nn.Embedding(n_actions, 32)

        # GRU sequence model: input = cat(z, a_embed), hidden = h
        self.gru = nn.GRUCell(z_dim + 32, h_dim)

        # Posterior: q(z_t | h_t, o_t)  — used during training with real obs
        self.posterior_net = nn.Sequential(
            nn.Linear(h_dim + 256, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, z_dim * 2),  # mu + log_sigma
        )

        # Prior: p(z_t | h_t)  — used during imagination (no obs)
        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, z_dim * 2),  # mu + log_sigma
        )

        # Reward predictor: r_t = f(h_t, z_t)
        self.reward_head = nn.Sequential(
            nn.Linear(h_dim + z_dim, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 1),
        )

        # Continuation predictor: c_t = f(h_t, z_t)  -> probability episode continues
        self.cont_head = nn.Sequential(
            nn.Linear(h_dim + z_dim, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 1),
        )

        # Decoder for visualization and reconstruction loss
        self.decoder = nn.Sequential(
            nn.Linear(h_dim + z_dim, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, int(np.prod(obs_shape))),
        )
        self._obs_shape = obs_shape

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def init_hidden(self, batch_size: int = 1, device=None):
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.h_dim, device=device)

    def _reparameterize(self, stats: torch.Tensor):
        """stats shape: (B, z_dim*2) -> sample z, return (z, mu, sigma)."""
        mu, log_sigma = stats.chunk(2, dim=-1)
        log_sigma = torch.clamp(log_sigma, -10, 2)
        sigma = log_sigma.exp()
        z = mu + sigma * torch.randn_like(sigma)
        return z, mu, sigma

    def observe_step(self, h: torch.Tensor, obs: torch.Tensor,
                     action: torch.Tensor, prev_z: torch.Tensor):
        """
        One step of RSSM with real observation (posterior mode).
        h:       (B, h_dim)
        obs:     (B, C, H, W)
        action:  (B,) long — action taken at previous step to arrive here
        prev_z:  (B, z_dim)
        Returns: h_new, z_post, mu_post, sigma_post, mu_prior, sigma_prior
        """
        a_emb = self.action_embed(action)                # (B, 32)
        gru_input = torch.cat([prev_z, a_emb], dim=-1)   # (B, z_dim+32)
        h_new = self.gru(gru_input, h)                   # (B, h_dim)

        # Prior from deterministic state only
        prior_stats = self.prior_net(h_new)
        _, mu_prior, sigma_prior = self._reparameterize(prior_stats)

        # Posterior from deterministic state + encoded obs
        enc_obs = self.obs_encoder(obs)                  # (B, 256)
        post_input = torch.cat([h_new, enc_obs], dim=-1) # (B, h_dim+256)
        post_stats = self.posterior_net(post_input)
        z_post, mu_post, sigma_post = self._reparameterize(post_stats)

        return h_new, z_post, mu_post, sigma_post, mu_prior, sigma_prior

    def imagine_step(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        """
        One step of RSSM in pure imagination (no obs, use prior).
        Returns: h_new, z_prior
        """
        a_emb = self.action_embed(action)
        gru_input = torch.cat([z, a_emb], dim=-1)
        h_new = self.gru(gru_input, h)
        prior_stats = self.prior_net(h_new)
        z_prior, _, _ = self._reparameterize(prior_stats)
        return h_new, z_prior

    def world_state(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, z], dim=-1)

    def predict_reward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.reward_head(self.world_state(h, z)).squeeze(-1)

    def predict_cont(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.cont_head(self.world_state(h, z)).squeeze(-1))

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        flat = self.decoder(self.world_state(h, z))
        return torch.sigmoid(flat).view(-1, *self._obs_shape)


class LatentActorCritic(nn.Module):
    """Actor and critic operating entirely in world-state latent space."""

    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 400),
            nn.ELU(inplace=True),
            nn.Linear(400, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def act(self, state: torch.Tensor, deterministic: bool = False):
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)


class SequenceReplayBuffer:
    """
    Stores complete episodes as sequences.
    Sampling returns fixed-length subsequences.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.episodes = collections.deque()
        self.total_stored = 0

    def push_episode(self, obs_list, act_list, rew_list, done_list):
        ep = {
            "obs":  np.stack(obs_list).astype(np.float32),    # (T, C, H, W)
            "acts": np.array(act_list, dtype=np.int64),        # (T,)
            "rews": np.array(rew_list, dtype=np.float32),      # (T,)
            "dones": np.array(done_list, dtype=np.float32),    # (T,)
        }
        self.episodes.append(ep)
        self.total_stored += len(act_list)
        # Remove oldest episodes until under capacity
        while self.total_stored > self.capacity and self.episodes:
            removed = self.episodes.popleft()
            self.total_stored -= len(removed["acts"])

    def sample(self, batch_size: int, seq_len: int, device: torch.device):
        obs_batch, act_batch, rew_batch, done_batch = [], [], [], []
        episodes = list(self.episodes)
        # Upweight episodes containing at least one nonzero reward 10x so the
        # reward head receives meaningful gradient despite sparse reward signal.
        weights = [1.0 + 9.0 * float(np.any(ep["rews"] != 0)) for ep in episodes]
        for _ in range(batch_size):
            ep = random.choices(episodes, weights=weights, k=1)[0]
            T = len(ep["acts"])
            if T <= seq_len:
                # Pad with last frame if episode shorter than seq_len
                start = 0
                pad = seq_len - T
            else:
                start = random.randint(0, T - seq_len)
                pad = 0

            end = min(start + seq_len, T)
            obs_  = ep["obs"][start:end]
            acts_ = ep["acts"][start:end]
            rews_ = ep["rews"][start:end]
            dones_= ep["dones"][start:end]

            if pad > 0:
                obs_  = np.concatenate([obs_,  np.tile(obs_[-1:],  (pad, 1, 1, 1))], 0)
                acts_ = np.concatenate([acts_, np.tile(acts_[-1:], (pad,))], 0)
                rews_ = np.concatenate([rews_, np.zeros(pad, dtype=np.float32)], 0)
                dones_= np.concatenate([dones_, np.ones(pad, dtype=np.float32)], 0)

            obs_batch.append(obs_)
            act_batch.append(acts_)
            rew_batch.append(rews_)
            done_batch.append(dones_)

        # Stack: (B, T, ...)
        obs_t  = torch.tensor(np.stack(obs_batch),  dtype=torch.float32, device=device)
        act_t  = torch.tensor(np.stack(act_batch),  dtype=torch.long,    device=device)
        rew_t  = torch.tensor(np.stack(rew_batch),  dtype=torch.float32, device=device)
        done_t = torch.tensor(np.stack(done_batch), dtype=torch.float32, device=device)
        return obs_t, act_t, rew_t, done_t

    def __len__(self):
        return len(self.episodes)


def compute_lambda_returns(rewards, conts, values, gamma: float = 0.99,
                           lam: float = 0.95):
    """
    Compute lambda-returns over imagined trajectory.
    rewards: list of (B,) tensors, length H
    conts:   list of (B,) tensors (continuation probability), length H
    values:  list of (B,) tensors, length H+1
    Returns: tensor (H, B)
    """
    # V_lambda_t = r_t + gamma*c_t*((1-lam)*V_{t+1} + lam*V_lambda_{t+1})
    # At t=H-1 (boundary): V_lambda_{H-1} = r_{H-1} + gamma*c_{H-1}*V_H
    H = len(rewards)
    returns = []
    for t in reversed(range(H)):
        if t == H - 1:
            lam_ret = rewards[t] + gamma * conts[t] * values[H]
        else:
            lam_ret = rewards[t] + gamma * conts[t] * ((1 - lam) * values[t + 1] + lam * returns[-1])
        returns.append(lam_ret)
    returns.reverse()
    return torch.stack(returns, dim=0)   # (H, B)


def _evaluate_dreamer(rssm: RSSM, ac: LatentActorCritic, env_key: str,
                      seed: int, z_dim: int,
                      n_episodes: int = 20, device=None) -> dict:
    """Dreamer-specific eval: resets RSSM state at each episode boundary."""
    if device is None:
        device = next(rssm.parameters()).device
    env = make_env(env_key, seed=seed + 10000)
    successes, ep_lengths, ep_returns = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        h = rssm.init_hidden(batch_size=1, device=device)
        z = torch.zeros(1, z_dim, device=device)
        prev_a = torch.zeros(1, dtype=torch.long, device=device)
        done = False
        ep_len = 0
        ep_ret = 0.0

        while not done:
            obs_t = obs_to_tensor(obs, device)
            with torch.no_grad():
                h, z_post, _, _, _, _ = rssm.observe_step(h, obs_t, prev_a, z)
                z = z_post
                state = rssm.world_state(h, z)
                action_t, _, _ = ac.act(state, deterministic=True)
                action = action_t.item()
                prev_a = action_t.unsqueeze(0) if action_t.dim() == 0 else action_t

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_len += 1
            ep_ret += float(reward)

        successes.append(1 if (terminated and ep_ret > 0) else 0)
        ep_lengths.append(ep_len)
        ep_returns.append(ep_ret)

    env.close()
    return {
        "success_rate": float(np.mean(successes)),
        "mean_ep_length": float(np.mean(ep_lengths)),
        "mean_return": float(np.mean(ep_returns)),
    }


def train_dreamer(env_key: str, seed: int, total_steps: int, save_dir: str,
                  device: torch.device, imagination_h: int = 15):
    set_seed(seed)
    env = make_env(env_key, seed)
    obs_shape = env.observation_space.shape    # (C, H, W)
    n_actions = env.action_space.n

    # Hyperparameters
    h_dim             = 512
    z_dim             = 32
    state_dim         = h_dim + z_dim          # 544
    imagination_H     = imagination_h
    seq_len           = 50
    wm_batch_size     = 50
    wm_lr             = 6e-4
    actor_lr          = 8e-5
    critic_lr         = 8e-5
    gamma             = 0.99
    lam               = 0.95
    kl_weight         = 1.0
    free_nats         = 1.0
    entropy_bonus     = 3e-4
    warmup_episodes   = 5        # episodes of random exploration before training
    eval_interval     = 5_000
    ckpt_interval     = 25_000

    rssm   = RSSM(obs_shape, n_actions, h_dim=h_dim, z_dim=z_dim).to(device)
    ac     = LatentActorCritic(state_dim=state_dim, n_actions=n_actions).to(device)
    seq_buf = SequenceReplayBuffer(capacity=100_000)

    wm_optimizer     = torch.optim.Adam(rssm.parameters(),  lr=wm_lr,    eps=1e-5)
    actor_optimizer  = torch.optim.Adam(ac.actor.parameters(),  lr=actor_lr,  eps=1e-5)
    critic_optimizer = torch.optim.Adam(ac.critic.parameters(), lr=critic_lr, eps=1e-5)

    log_suffix = f"_h{imagination_H}" if imagination_H != 15 else ""
    log_path = os.path.join(save_dir, "logs", f"dreamer_{env_key}_seed{seed}{log_suffix}.csv")
    logger = CSVLogger(log_path)

    global_step = 0
    episode_count = 0
    wm_loss_ema = None
    recon_loss_ema = None
    kl_loss_ema = None
    actor_loss_ema = None
    critic_loss_ema = None
    wm_grad_norm_ema = None
    last_eval_step = -eval_interval
    last_ckpt_step = -ckpt_interval

    print(f"[Dreamer] env={env_key} seed={seed} total_steps={total_steps} H={imagination_H}")

    while global_step < total_steps:

        # ==================================================================
        # Phase 1: Collect one episode from the real environment
        # ==================================================================
        obs, _ = env.reset()
        ep_obs, ep_acts, ep_rews, ep_dones = [], [], [], []
        done = False

        # For acting with trained actor, maintain running RSSM state
        h = rssm.init_hidden(batch_size=1, device=device)
        z = torch.zeros(1, z_dim, device=device)
        prev_action = torch.zeros(1, dtype=torch.long, device=device)

        while not done:
            with torch.no_grad():
                obs_t = obs_to_tensor(obs, device)
                h_new, z_post, _, _, _, _ = rssm.observe_step(h, obs_t, prev_action, z)
                h, z = h_new, z_post
                state = rssm.world_state(h, z)

                if episode_count < warmup_episodes:
                    action = env.action_space.sample()
                else:
                    action_t, _, _ = ac.act(state, deterministic=False)
                    action = action_t.item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ep_obs.append(obs)
            ep_acts.append(action)
            ep_rews.append(reward)
            ep_dones.append(float(done))

            obs = next_obs
            prev_action = torch.tensor([action], dtype=torch.long, device=device)
            global_step += 1

        seq_buf.push_episode(ep_obs, ep_acts, ep_rews, ep_dones)
        episode_count += 1

        if len(seq_buf) < warmup_episodes:
            continue

        # ==================================================================
        # Phase 2: World Model Training
        # ==================================================================
        rssm.train()
        obs_b, act_b, rew_b, done_b = seq_buf.sample(wm_batch_size, seq_len, device)
        # obs_b:  (B, T, C, H, W)
        # act_b:  (B, T)  — action taken to GET to this obs (previous action)
        # rew_b:  (B, T)
        # done_b: (B, T)

        B, T = act_b.shape

        h_t = rssm.init_hidden(batch_size=B, device=device)
        z_t = torch.zeros(B, z_dim, device=device)
        # First action is zeros (no previous action at episode start)
        a_prev = torch.zeros(B, dtype=torch.long, device=device)

        total_recon_loss  = torch.tensor(0.0, device=device)
        total_kl_loss     = torch.tensor(0.0, device=device)
        total_reward_loss = torch.tensor(0.0, device=device)
        total_cont_loss   = torch.tensor(0.0, device=device)

        for t in range(T):
            obs_t_step = obs_b[:, t]          # (B, C, H, W)
            a_t        = act_b[:, t]           # (B,) — action at this step
            r_t        = rew_b[:, t]           # (B,)
            d_t        = done_b[:, t]          # (B,)

            h_t, z_post, mu_post, sigma_post, mu_prior, sigma_prior = \
                rssm.observe_step(h_t, obs_t_step, a_prev, z_t)

            # Reconstruction loss (decoder)
            recon = rssm.decode(h_t, z_post)
            recon_loss = F.mse_loss(recon, obs_t_step)

            # KL divergence: q(z|h,o) || p(z|h)
            # KL of two gaussians: 0.5*(sigma_post/sigma_prior)^2
            #   + ((mu_post-mu_prior)/sigma_prior)^2 - 1 + 2*log(sigma_prior/sigma_post)
            var_post  = sigma_post.pow(2)
            var_prior = sigma_prior.pow(2)
            kl = 0.5 * (
                var_post / (var_prior + 1e-8)
                + (mu_post - mu_prior).pow(2) / (var_prior + 1e-8)
                - 1.0
                + (var_prior + 1e-8).log() - (var_post + 1e-8).log()
            ).sum(dim=-1)
            kl = torch.clamp(kl, min=free_nats).mean()

            # Reward prediction loss
            pred_r = rssm.predict_reward(h_t, z_post)
            reward_loss = F.mse_loss(pred_r, r_t)

            # Continuation prediction loss (1 = episode continues)
            pred_c = rssm.predict_cont(h_t, z_post)
            cont_loss = F.binary_cross_entropy(pred_c, 1.0 - d_t)

            total_recon_loss  = total_recon_loss  + recon_loss
            total_kl_loss     = total_kl_loss     + kl
            total_reward_loss = total_reward_loss + reward_loss
            total_cont_loss   = total_cont_loss   + cont_loss

            z_t    = z_post
            a_prev = a_t

        total_recon_loss  = total_recon_loss  / T
        total_kl_loss     = total_kl_loss     / T
        total_reward_loss = total_reward_loss / T
        total_cont_loss   = total_cont_loss   / T

        wm_loss = (total_recon_loss
                   + kl_weight * total_kl_loss
                   + total_reward_loss
                   + total_cont_loss)

        wm_optimizer.zero_grad()
        wm_loss.backward()
        gn_wm = nn.utils.clip_grad_norm_(rssm.parameters(), max_norm=100.0).item()
        wm_optimizer.step()

        wm_loss_ema    = wm_loss.item()               if wm_loss_ema    is None else 0.99 * wm_loss_ema    + 0.01 * wm_loss.item()
        recon_loss_ema = total_recon_loss.item()      if recon_loss_ema is None else 0.99 * recon_loss_ema + 0.01 * total_recon_loss.item()
        kl_loss_ema    = total_kl_loss.item()         if kl_loss_ema    is None else 0.99 * kl_loss_ema    + 0.01 * total_kl_loss.item()
        wm_grad_norm_ema = gn_wm if wm_grad_norm_ema is None else 0.99 * wm_grad_norm_ema + 0.01 * gn_wm

        # ==================================================================
        # Phase 3: Actor-Critic Training on Imagined Rollouts
        # ==================================================================
        if episode_count >= warmup_episodes:
            rssm.eval()
            ac.train()

            # Sample starting states from real encoded posteriors
            obs_start, _, _, _ = seq_buf.sample(wm_batch_size, seq_len=1, device=device)
            with torch.no_grad():
                h_start = rssm.init_hidden(batch_size=wm_batch_size, device=device)
                z_start = torch.zeros(wm_batch_size, z_dim, device=device)
                a_init  = torch.zeros(wm_batch_size, dtype=torch.long, device=device)
                h_start, z_start, _, _, _, _ = rssm.observe_step(
                    h_start, obs_start[:, 0], a_init, z_start)

            # Imagine H steps forward in latent space — NO env interaction.
            # Collect states, rewards, conts, and actor log_probs in one pass.
            imag_hs       = [h_start.detach()]
            imag_zs       = [z_start.detach()]
            imag_rews     = []
            imag_conts    = []
            imag_log_prob = []
            imag_entropy  = []

            h_im = h_start.detach()
            z_im = z_start.detach()

            for step_h in range(imagination_H):
                state_im = rssm.world_state(h_im, z_im)       # (B, 544)
                action_im, lp_im, ent_im = ac.act(state_im)   # gradients kept for actor

                with torch.no_grad():
                    h_im_new, z_im_new = rssm.imagine_step(h_im, z_im, action_im)
                    r_im = rssm.predict_reward(h_im_new, z_im_new)
                    c_im = rssm.predict_cont(h_im_new, z_im_new)

                imag_hs.append(h_im_new)
                imag_zs.append(z_im_new)
                imag_rews.append(r_im)
                imag_conts.append(c_im)
                imag_log_prob.append(lp_im)
                imag_entropy.append(ent_im)

                h_im = h_im_new
                z_im = z_im_new

            # Compute values at all H+1 states
            with torch.no_grad():
                imag_values = []
                for i in range(imagination_H + 1):
                    s_i = rssm.world_state(imag_hs[i], imag_zs[i])
                    imag_values.append(ac.value(s_i).detach())

            # Lambda-returns: (H, B)
            lambda_rets = compute_lambda_returns(
                imag_rews, imag_conts, imag_values, gamma=gamma, lam=lam
            )

            # Actor loss: REINFORCE with lambda-return baseline
            actor_loss_total = torch.tensor(0.0, device=device)
            entropy_total    = torch.tensor(0.0, device=device)
            for step_h in range(imagination_H):
                actor_loss_total = actor_loss_total - (
                    lambda_rets[step_h].detach() * imag_log_prob[step_h]).mean()
                entropy_total = entropy_total + imag_entropy[step_h].mean()

            actor_loss = actor_loss_total / imagination_H
            actor_loss = actor_loss - entropy_bonus * (entropy_total / imagination_H)

            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(ac.actor.parameters(), max_norm=100.0)
            actor_optimizer.step()

            # Critic loss: regress to lambda-returns
            critic_loss_total = torch.tensor(0.0, device=device)
            for step_h in range(imagination_H):
                s_i = rssm.world_state(imag_hs[step_h].detach(),
                                       imag_zs[step_h].detach())
                v_pred = ac.value(s_i)
                critic_loss_total = critic_loss_total + F.mse_loss(
                    v_pred, lambda_rets[step_h].detach())
            critic_loss = critic_loss_total / imagination_H

            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(ac.critic.parameters(), max_norm=100.0)
            critic_optimizer.step()

            actor_loss_ema  = actor_loss.item()  if actor_loss_ema  is None else 0.99 * actor_loss_ema  + 0.01 * actor_loss.item()
            critic_loss_ema = critic_loss.item() if critic_loss_ema is None else 0.99 * critic_loss_ema + 0.01 * critic_loss.item()
            rssm.train()

        # ---- Eval + log ----
        if global_step - last_eval_step >= eval_interval:
            last_eval_step = global_step
            rssm.eval(); ac.eval()
            eval_results = _evaluate_dreamer(rssm, ac, env_key, seed,
                                             z_dim, n_episodes=20, device=device)
            row = {
                "step": global_step,
                "success_rate": eval_results["success_rate"],
                "mean_ep_length": eval_results["mean_ep_length"],
                "mean_return": eval_results["mean_return"],
                "wm_loss": wm_loss_ema if wm_loss_ema is not None else 0.0,
                "actor_loss": actor_loss_ema if actor_loss_ema is not None else 0.0,
                "critic_loss": critic_loss_ema if critic_loss_ema is not None else 0.0,
                "recon_loss": recon_loss_ema if recon_loss_ema is not None else 0.0,
                "kl_loss":   kl_loss_ema   if kl_loss_ema   is not None else 0.0,
                "grad_norm": wm_grad_norm_ema if wm_grad_norm_ema is not None else 0.0,
            }
            logger.write(row)
            _wm  = wm_loss_ema    if wm_loss_ema    is not None else 0.0
            _al  = actor_loss_ema if actor_loss_ema is not None else 0.0
            print(f"  step={global_step:>7d}  success={eval_results['success_rate']:.3f}"
                  f"  wm_loss={_wm:.4f}  a_loss={_al:.4f}")
            rssm.train(); ac.train()

        if global_step - last_ckpt_step >= ckpt_interval:
            last_ckpt_step = global_step
            ckpt_path = os.path.join(save_dir, "checkpoints",
                                     f"dreamer_{env_key}_seed{seed}_step{global_step}.pt")
            save_checkpoint({
                "rssm":   rssm.state_dict(),
                "actor":  ac.actor.state_dict(),
                "critic": ac.critic.state_dict(),
                "step":   global_step,
            }, ckpt_path)

    logger.close()
    env.close()
    print(f"[Dreamer] Training complete. Logs: {log_path}")


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="RL Benchmarking: PPO vs Dyna-Q vs Dreamer")
    parser.add_argument("--agent",       type=str, required=True,
                        choices=["ppo", "dynaq", "dreamer"],
                        help="Which agent to train.")
    parser.add_argument("--env",         type=str, required=True,
                        choices=list(ENV_MAP.keys()),
                        help="Which MiniGrid environment.")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--save_dir",    type=str, default="results")
    parser.add_argument("--device",      type=str, default="auto",
                        help="'auto', 'cpu', or 'cuda'.")
    parser.add_argument("--planning_k",   type=int, default=None,
                        help="Dyna-Q planning steps per env step (default 5).")
    parser.add_argument("--imagination_h", type=int, default=None,
                        help="Dreamer imagination horizon in steps (default 15).")
    parser.add_argument("--eval_generalization", action="store_true",
                        help="After training, eval the final checkpoint zero-shot on the 16x16 env variant.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Agent: {args.agent} | Env: {args.env} | Seed: {args.seed}"
          f" | Steps: {args.total_steps}")

    os.makedirs(os.path.join(args.save_dir, "logs"),        exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "tables"),      exist_ok=True)

    if args.agent == "ppo":
        train_ppo(args.env, args.seed, args.total_steps, args.save_dir, device)
    elif args.agent == "dynaq":
        train_dynaq(args.env, args.seed, args.total_steps, args.save_dir, device,
                    planning_k=args.planning_k if args.planning_k is not None else 5)
    elif args.agent == "dreamer":
        train_dreamer(args.env, args.seed, args.total_steps, args.save_dir, device,
                      imagination_h=args.imagination_h if args.imagination_h is not None else 15)

    # Zero-shot generalization eval on 16x16 variant
    if args.eval_generalization:
        env_tmp = make_env(args.env, seed=0)
        obs_shape = env_tmp.observation_space.shape
        n_actions = env_tmp.action_space.n
        env_tmp.close()

        ckpt_path = _find_latest_checkpoint(args.save_dir, args.agent, args.env, args.seed)
        if ckpt_path is None:
            print("[gen_eval] No checkpoint found — skipping generalization eval.")
        else:
            print(f"[gen_eval] Evaluating {ckpt_path} on 16x16 env...")
            gen_rate = load_checkpoint_and_eval_generalization(
                args.agent, ckpt_path, args.env, obs_shape, n_actions, device)
            gen_path = os.path.join(args.save_dir, "tables", "generalization_results.csv")
            _append_gen_result(gen_path, args.agent, args.env, args.seed, gen_rate)
            print(f"[gen_eval] {args.agent} on {args.env} 16x16: success={gen_rate:.3f}"
                  f"  → {gen_path}")


if __name__ == "__main__":
    main()
