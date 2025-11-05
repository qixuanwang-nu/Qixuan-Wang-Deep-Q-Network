#!/usr/bin/env python3
import os
import math
import json
import time
import random
import argparse
from collections import deque, namedtuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import gymnasium as gym
except ImportError:
    import gym  # type: ignore

import ale_py
# Register ALE environments
gym.register_envs(ale_py)
# -----------------------------
# Preprocessing (as specified)
# -----------------------------
mspacman_color = 210 + 164 + 74


def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    """
    Provided preprocessing for MsPacman.
    Returns HxWx1 int8 image (88x80x1) with values in [-128, 127].
    """
    img = obs[1:176:2, ::2]  # crop and downsize
    img = img.sum(axis=2)  # to greyscale
    img[img == mspacman_color] = 0  # Improve contrast
    img = (img // 3 - 128).astype(np.int8)  # normalize from -128 to 127
    return img.reshape(88, 80, 1)


# -----------------------------
# Utilities
# -----------------------------


def make_env(seed: int | None = None):
    """Create the MsPacman environment with Gymnasium API.

    Tries 'ALE/MsPacman-v5' first (Gymnasium Atari), then 'MsPacman-v5'.
    """
    env = None
    last_err = None
    for env_id in ["ALE/MsPacman-v5", "MsPacman-v5"]:
        try:
            env = gym.make(env_id, render_mode=None)  # type: ignore[arg-type]
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    if env is None:
        raise RuntimeError(
            f"Failed to create MsPacman env. Last error: {last_err}"
        )

    if seed is not None:
        try:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        except TypeError:
            # older gym versions
            pass
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    return env


class FrameStacker:
    """Maintain a stack of the most recent k preprocessed frames.

    Stores frames as int8 and concatenates along channel (C) dimension for PyTorch.
    Output shape: (k, 88, 80)
    """

    def __init__(self, k: int):
        assert k >= 1
        self.k = k
        self.frames: deque[np.ndarray] = deque(maxlen=k)

    def reset(self):
        self.frames.clear()

    def push(self, frame_hwc: np.ndarray):
        # Convert HWC (88,80,1) int8 -> (1,88,80)
        c = np.moveaxis(frame_hwc, -1, 0)
        self.frames.append(c)

    def get_state(self) -> np.ndarray:
        if len(self.frames) == 0:
            # should not happen if used correctly
            zero = np.zeros((1, 88, 80), dtype=np.int8)
            return np.repeat(zero, self.k, axis=0)
        if len(self.frames) < self.k:
            first = self.frames[0]
            # pad with first frame
            pads = [first] * (self.k - len(self.frames))
            stacked = np.concatenate(list(self.frames) + pads, axis=0)
        else:
            stacked = np.concatenate(list(self.frames), axis=0)
        return stacked.astype(np.int8)


# -----------------------------
# Replay Buffer
# -----------------------------


Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: tuple[int, int, int]):
        # state_shape: (C, H, W)
        self.capacity = int(capacity)
        self.size = 0
        self.pos = 0

        C, H, W = state_shape
        # Store as uint8/int8 to save RAM; convert to float32 on batch
        self.states = np.zeros((self.capacity, C, H, W), dtype=np.int8)
        self.actions = np.zeros((self.capacity,), dtype=np.int16)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, C, H, W), dtype=np.int8)
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size: int, device: torch.device, beta: float | None = None):
        idx = np.random.randint(0, self.size, size=batch_size)
        states = torch.from_numpy(self.states[idx].astype(np.int8).astype(np.float32) / 128.0).to(device)
        next_states = torch.from_numpy(self.next_states[idx].astype(np.int8).astype(np.float32) / 128.0).to(device)
        actions = torch.from_numpy(self.actions[idx].astype(np.int64)).to(device)
        rewards = torch.from_numpy(self.rewards[idx].astype(np.float32)).to(device)
        dones = torch.from_numpy(self.dones[idx].astype(np.float32)).to(device)
        weights = torch.ones(batch_size, device=device, dtype=torch.float32)
        return states, actions, rewards, next_states, dones, weights, idx


# -----------------------------
# Prioritized Replay Buffer (Proportional PER)
# -----------------------------


class SumSegmentTree:
    def __init__(self, capacity: int):
        assert capacity > 0 and (capacity & (capacity - 1) == 0), "capacity must be power of 2"
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)

    def sum(self) -> float:
        return float(self.tree[1])

    def add(self, idx: int, value: float):
        i = idx + self.capacity
        self.tree[i] = value
        i //= 2
        while i >= 1:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            i //= 2

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        i = 1
        while i < self.capacity:
            left = 2 * i
            if self.tree[left] >= prefixsum:
                i = left
            else:
                prefixsum -= self.tree[left]
                i = left + 1
        return i - self.capacity


class MinSegmentTree:
    def __init__(self, capacity: int):
        assert capacity > 0 and (capacity & (capacity - 1) == 0), "capacity must be power of 2"
        self.capacity = capacity
        self.tree = np.full(2 * capacity, np.inf, dtype=np.float32)

    def min(self) -> float:
        return float(self.tree[1])

    def add(self, idx: int, value: float):
        i = idx + self.capacity
        self.tree[i] = value
        i //= 2
        while i >= 1:
            self.tree[i] = min(self.tree[2 * i], self.tree[2 * i + 1])
            i //= 2


def _next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length()


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, state_shape: tuple[int, int, int], alpha: float = 0.6, eps: float = 1e-6):
        super().__init__(capacity, state_shape)
        self.alpha = float(alpha)
        self.eps = float(eps)
        tree_capacity = _next_power_of_two(self.capacity)
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.max_priority = 1.0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        super().push(state, action, reward, next_state, done)
        idx = (self.pos - 1) % self.capacity
        p = self.max_priority ** self.alpha
        self.sum_tree.add(idx, p)
        self.min_tree.add(idx, p)

    def sample(self, batch_size: int, device: torch.device, beta: float | None = None):
        assert self.size > 0
        if beta is None:
            beta = 0.4
        total = self.sum_tree.sum()
        segment = total / batch_size if batch_size > 0 else total
        idxes = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.random() * (b - a) + a
            idx = self.sum_tree.find_prefixsum_idx(s)
            if idx >= self.size:
                idx = random.randrange(self.size)
            idxes.append(idx)
        idx = np.array(idxes, dtype=np.int64)

        states = torch.from_numpy(self.states[idx].astype(np.int8).astype(np.float32) / 128.0).to(device)
        next_states = torch.from_numpy(self.next_states[idx].astype(np.int8).astype(np.float32) / 128.0).to(device)
        actions = torch.from_numpy(self.actions[idx].astype(np.int64)).to(device)
        rewards = torch.from_numpy(self.rewards[idx].astype(np.float32)).to(device)
        dones = torch.from_numpy(self.dones[idx].astype(np.float32)).to(device)

        priorities = np.array([self.sum_tree.tree[self.sum_tree.capacity + int(i)] for i in idx], dtype=np.float32)
        probs = priorities / (total + 1e-12)
        min_prob = self.min_tree.min() / (total + 1e-12) if total > 0 else 1.0
        weights = (probs / (min_prob + 1e-12)) ** (-beta)
        weights = torch.from_numpy(weights.astype(np.float32)).to(device)
        return states, actions, rewards, next_states, dones, weights, idx

    def update_priorities(self, idx: np.ndarray, priorities: np.ndarray):
        priorities = np.asarray(priorities, dtype=np.float32) + self.eps
        for i, p in zip(idx, priorities):
            prio = float(p) ** self.alpha
            self.sum_tree.add(int(i), prio)
            self.min_tree.add(int(i), prio)
            self.max_priority = max(self.max_priority, float(p))


# -----------------------------
# DQN Network
# -----------------------------


class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet layer (Fortunato et al., 2018)."""

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Buffers for noise
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        mu_range = 1.0 / math.sqrt(in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma0 / math.sqrt(in_features))
        self.bias_sigma.data.fill_(sigma0 / math.sqrt(in_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features).to(self.weight_mu.device)
        eps_out = self._scale_noise(self.out_features).to(self.weight_mu.device)
        self.weight_epsilon = eps_out.ger(eps_in)
        self.bias_epsilon = eps_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)


class DQN(nn.Module):
    def __init__(self, in_channels: int, n_actions: int, *, dueling: bool = False, noisy: bool = False, hidden_dim: int = 512):
        super().__init__()
        self.dueling = dueling
        self.noisy = noisy
        self.hidden_dim = hidden_dim
        # Nature DQN-like, adapted to input 88x80
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        # Compute conv output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 88, 80)
            n_flat = self.features(dummy).view(1, -1).size(1)
        self.n_flat = n_flat

        if not dueling:
            if noisy:
                self.head = nn.Sequential(
                    nn.Linear(n_flat, hidden_dim),
                    nn.ReLU(inplace=True),
                    NoisyLinear(hidden_dim, n_actions),
                )
            else:
                self.head = nn.Sequential(
                    nn.Linear(n_flat, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, n_actions),
                )
        else:
            if noisy:
                self.value_stream = nn.Sequential(
                    nn.Linear(n_flat, hidden_dim),
                    nn.ReLU(inplace=True),
                    NoisyLinear(hidden_dim, 1),
                )
                self.adv_stream = nn.Sequential(
                    nn.Linear(n_flat, hidden_dim),
                    nn.ReLU(inplace=True),
                    NoisyLinear(hidden_dim, n_actions),
                )
            else:
                self.value_stream = nn.Sequential(
                    nn.Linear(n_flat, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, 1),
                )
                self.adv_stream = nn.Sequential(
                    nn.Linear(n_flat, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, n_actions),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        if not self.dueling:
            return self.head(x)
        v = self.value_stream(x)
        a = self.adv_stream(x)
        return v + a - a.mean(dim=1, keepdim=True)


def build_q_network(in_channels: int, n_actions: int, args) -> DQN:
    return DQN(
        in_channels=in_channels,
        n_actions=n_actions,
        dueling=getattr(args, "dueling", False),
        noisy=getattr(args, "noisy_net", False),
        hidden_dim=getattr(args, "hidden_dim", 512),
    )


# -----------------------------
# Epsilon scheduler
# -----------------------------


class LinearEpsilonScheduler:
    def __init__(self, start: float, end: float, decay_steps: int):
        self.start = float(start)
        self.end = float(end)
        self.decay_steps = max(1, int(decay_steps))

    def value(self, step: int) -> float:
        if step >= self.decay_steps:
            return self.end
        frac = step / self.decay_steps
        return self.start + (self.end - self.start) * frac


# -----------------------------
# Training Loop
# -----------------------------


def select_action(q_net: DQN, state: np.ndarray, epsilon: float, device: torch.device, n_actions: int) -> tuple[int, float]:
    """
    Epsilon-greedy action selection. Returns (action, max_q_estimate).
    state: np.ndarray, shape (C, 88, 80), int8 in [-128,127]
    """
    if random.random() < epsilon:
        action = random.randrange(n_actions)
        # Still compute Q for logging stability (optional)
        with torch.no_grad():
            s = torch.from_numpy(state.astype(np.float32) / 128.0).unsqueeze(0).to(device)
            q = q_net(s)
            max_q = float(q.max().item())
        return action, max_q
    else:
        with torch.no_grad():
            s = torch.from_numpy(state.astype(np.float32) / 128.0).unsqueeze(0).to(device)
            q = q_net(s)
            action = int(q.argmax(dim=1).item())
            max_q = float(q.max().item())
        return action, max_q


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)


def hard_update(target: nn.Module, source: nn.Module):
    target.load_state_dict(source.state_dict())


def compute_td_target(
    reward: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    next_q_values: torch.Tensor,
) -> torch.Tensor:
    # done is 1.0 if terminal, 0.0 otherwise
    return reward + (1.0 - done) * gamma * next_q_values


def train(
    args,
):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "logs"), exist_ok=True)

    env = make_env(seed=args.seed)
    n_actions = env.action_space.n
    assert n_actions == 9, f"Expected 9 actions for MsPacman, got {n_actions}"

    frame_stacker = FrameStacker(args.frame_stack)

    # Networks
    q_net = build_q_network(in_channels=args.frame_stack, n_actions=n_actions, args=args).to(device)
    target_net = build_q_network(in_channels=args.frame_stack, n_actions=n_actions, args=args).to(device)
    hard_update(target_net, q_net)
    target_net.eval()

    # Optimizer
    if args.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(q_net.parameters(), lr=args.learning_rate, alpha=0.95, eps=0.00001)
    else:
        optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    loss_fn = nn.SmoothL1Loss(reduction="none")  # Huber

    # Replay buffer
    if getattr(args, "per", False):
        buffer = PrioritizedReplayBuffer(
            capacity=args.replay_size,
            state_shape=(args.frame_stack, 88, 80),
            alpha=args.per_alpha,
            eps=args.per_eps,
        )
    else:
        buffer = ReplayBuffer(
            capacity=args.replay_size,
            state_shape=(args.frame_stack, 88, 80),
        )

    eps_sched = LinearEpsilonScheduler(
        start=args.epsilon_start,
        end=args.epsilon_end,
        decay_steps=args.epsilon_decay_steps,
    )

    global_step = 0
    episode_rewards: list[float] = []
    episode_max_qs: list[float] = []
    moving_avg_100: list[float] = []

    # CSV log
    log_path = os.path.join(args.outdir, "logs", "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("episode,steps,reward,epsilon,max_q,avg100,lr\n")

    for ep in range(1, args.train_episodes + 1):
        obs, info = env.reset()
        frame_stacker.reset()
        f = preprocess_observation(obs)
        for _ in range(args.frame_stack):
            frame_stacker.push(f)
        state = frame_stacker.get_state()

        prev_lives = None
        try:
            if isinstance(info, dict) and "lives" in info:
                prev_lives = int(info["lives"])  # provided by Gymnasium Atari
        except Exception:
            prev_lives = None

        ep_reward = 0.0
        ep_max_q = -float("inf")
        ep_steps = 0

        # N-step accumulation queue
        nstep_queue: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=args.n_step)

        for t in range(args.max_steps_per_episode):
            epsilon = eps_sched.value(global_step)
            action, max_q = select_action(q_net, state, epsilon, device, n_actions)
            ep_max_q = max(ep_max_q, max_q)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)

            # Training reward (optional clipping)
            train_reward = float(np.clip(reward, -args.reward_clip, args.reward_clip)) if args.reward_clip is not None and args.reward_clip > 0 else float(reward)

            # Life-loss terminal used only for training targets
            if args.terminal_on_life_loss:
                try:
                    lives = int(info.get("lives", prev_lives)) if isinstance(info, dict) else prev_lives
                except Exception:
                    lives = prev_lives
                life_lost = prev_lives is not None and lives is not None and lives < prev_lives
                prev_lives = lives
            else:
                life_lost = False

            done_for_buffer = bool(done or life_lost)

            f2 = preprocess_observation(next_obs)
            frame_stacker.push(f2)
            next_state = frame_stacker.get_state()

            # n-step accumulation
            nstep_queue.append((state, action, train_reward, next_state, done_for_buffer))
            if len(nstep_queue) == args.n_step:
                R = 0.0
                gamma_acc = 1.0
                done_acc = False
                for (_s, _a, r_i, _ns, d_i) in nstep_queue:
                    R += gamma_acc * r_i
                    gamma_acc *= args.gamma
                    if d_i:
                        done_acc = True
                        break
                s0, a0, _r0, _ns0, _d0 = nstep_queue[0]
                ns_last = nstep_queue[-1][3]
                buffer.push(s0, a0, R, ns_last, done_acc)

            state = next_state
            global_step += 1
            ep_steps += 1

            # Learn
            if (
                len(buffer) >= args.learning_starts
                and global_step % args.train_freq == 0
            ):
                beta = 1.0
                if getattr(args, "per", False):
                    beta = min(1.0, args.per_beta0 + (1.0 - args.per_beta0) * (global_step / max(1, args.per_beta_steps)))
                states, actions, rewards, next_states, dones, weights, idx = buffer.sample(
                    args.batch_size, device, beta=beta
                )
                q_values = q_net(states)
                q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    if args.double_dqn:
                        # Double DQN: action selection by online, eval by target
                        next_q_online = q_net(next_states)
                        next_actions = next_q_online.argmax(dim=1, keepdim=True)
                        next_q_target = target_net(next_states)
                        next_q_max = next_q_target.gather(1, next_actions).squeeze(1)
                    else:
                        next_q_target = target_net(next_states)
                        next_q_max, _ = next_q_target.max(dim=1)
                    gamma_n = args.gamma ** args.n_step
                    td_target = compute_td_target(
                        rewards, dones, gamma_n, next_q_max
                    )

                loss_elts = loss_fn(q_selected, td_target)
                if weights is not None:
                    loss = (loss_elts * weights).mean()
                else:
                    loss = loss_elts.mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                # PER priority updates
                if getattr(args, "per", False):
                    with torch.no_grad():
                        td_errors = (td_target - q_selected).abs().detach().cpu().numpy()
                    buffer.update_priorities(idx, td_errors)

            # Target update
            if global_step % args.target_update_freq == 0:
                if args.tau > 0.0:
                    soft_update(target_net, q_net, args.tau)
                else:
                    hard_update(target_net, q_net)

            if done:
                # Flush remaining n-step transitions
                while len(nstep_queue) > 0:
                    R = 0.0
                    gamma_acc = 1.0
                    done_acc = False
                    for (_s, _a, r_i, _ns, d_i) in nstep_queue:
                        R += gamma_acc * r_i
                        gamma_acc *= args.gamma
                        if d_i:
                            done_acc = True
                            break
                    s0, a0, _r0, _ns0, _d0 = nstep_queue[0]
                    ns_last = nstep_queue[-1][3]
                    buffer.push(s0, a0, R, ns_last, done_acc)
                    nstep_queue.popleft()
                break

        episode_rewards.append(ep_reward)
        episode_max_qs.append(ep_max_q if ep_max_q != -float("inf") else 0.0)
        avg100 = float(np.mean(episode_rewards[-100:]))
        moving_avg_100.append(avg100)

        # Log
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{ep},{ep_steps},{ep_reward:.3f},{epsilon:.6f},{episode_max_qs[-1]:.5f},{avg100:.3f},{scheduler.get_last_lr()[0]:.8f}\n"
            )

        # Periodic checkpoint
        if ep % args.checkpoint_interval == 0 or ep == args.train_episodes:
            ckpt = {
                "model_state": q_net.state_dict(),
                "target_state": target_net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "args": vars(args),
                "global_step": global_step,
                "episode": ep,
            }
            torch.save(
                ckpt,
                os.path.join(args.outdir, "checkpoints", f"dqn_ep{ep}_step{global_step}.pt"),
            )

        # Console status
        if ep % max(1, args.status_interval) == 0:
            print(
                f"Episode {ep}/{args.train_episodes} | Reward {ep_reward:.1f} | Avg100 {avg100:.1f} | MaxQ {episode_max_qs[-1]:.3f} | Eps {epsilon:.3f} | Steps {global_step}"
            )

    env.close()

    # Save training summary
    summary = {
        "episodes": args.train_episodes,
        "rewards": episode_rewards,
        "max_qs": episode_max_qs,
        "moving_avg_100": moving_avg_100,
    }
    with open(os.path.join(args.outdir, "logs", "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f)

    # Plots
    plot_training_curves(episode_rewards, moving_avg_100, episode_max_qs, args.outdir)


def plot_training_curves(rewards, avg100, max_qs, outdir: str):
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # (i) Max Q-values vs episodes
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(max_qs) + 1), max_qs, label="Max Q per episode")
    plt.xlabel("Episode")
    plt.ylabel("Max Q-value")
    plt.title("Maximum Q-values vs Episodes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "max_q_values.png"), dpi=150)
    plt.close()

    # (ii) Rewards vs episodes with 100-ep moving average
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(rewards) + 1), rewards, label="Episode Reward", alpha=0.6)
    plt.plot(range(1, len(avg100) + 1), avg100, label="Moving Avg (100)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Rewards and Moving Average (100)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "episode_rewards.png"), dpi=150)
    plt.close()


# -----------------------------
# Evaluation
# -----------------------------


def load_checkpoint(model_path: str, frame_stack: int, n_actions: int, device: torch.device):
    data = torch.load(model_path, map_location=device)
    # Rebuild network with saved architecture flags if present
    ckpt_args = data.get("args", {})
    class _Dummy:
        pass
    dummy = _Dummy()
    for k, v in ckpt_args.items():
        setattr(dummy, k.replace('-', '_'), v)
    if not hasattr(dummy, "dueling"):
        dummy.dueling = False
    if not hasattr(dummy, "noisy_net"):
        dummy.noisy_net = False
    if not hasattr(dummy, "hidden_dim"):
        dummy.hidden_dim = 512
    q_net = build_q_network(in_channels=frame_stack, n_actions=n_actions, args=dummy).to(device)
    state_dict = data.get("model_state", data)
    q_net.load_state_dict(state_dict)
    q_net.eval()
    return q_net


def evaluate(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "plots"), exist_ok=True)

    env = make_env(seed=args.seed)
    n_actions = env.action_space.n

    # Resolve checkpoint path
    model_path = args.model
    if model_path is None:
        # try latest from checkpoints
        ckpt_dir = os.path.join(args.outdir, "checkpoints")
        cands = []
        if os.path.isdir(ckpt_dir):
            for fn in os.listdir(ckpt_dir):
                if fn.endswith(".pt"):
                    cands.append(os.path.join(ckpt_dir, fn))
        if not cands:
            raise FileNotFoundError("No checkpoint found. Provide --model <path>.")
        model_path = sorted(cands)[-1]
        print(f"Using latest checkpoint: {model_path}")

    q_net = load_checkpoint(model_path, frame_stack=args.frame_stack, n_actions=n_actions, device=device)

    rewards = []
    frame_stacker = FrameStacker(args.frame_stack)

    for ep in range(1, args.eval_episodes + 1):
        obs, _ = env.reset()
        frame_stacker.reset()
        f = preprocess_observation(obs)
        for _ in range(args.frame_stack):
            frame_stacker.push(f)
        state = frame_stacker.get_state()

        ep_reward = 0.0
        for t in range(args.max_steps_per_episode):
            with torch.no_grad():
                s = torch.from_numpy(state.astype(np.float32) / 128.0).unsqueeze(0).to(device)
                q = q_net(s)
                action = int(q.argmax(dim=1).item())
            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)

            f2 = preprocess_observation(next_obs)
            frame_stacker.push(f2)
            state = frame_stacker.get_state()
            if done:
                break
        rewards.append(ep_reward)

        if ep % max(1, args.status_interval) == 0:
            avg100 = np.mean(rewards[-100:])
            print(f"Eval Episode {ep}/{args.eval_episodes} | Reward {ep_reward:.1f} | Avg100 {avg100:.1f}")

    env.close()

    rewards_np = np.array(rewards, dtype=np.float32)
    mean_r = float(rewards_np.mean()) if len(rewards_np) > 0 else 0.0
    std_r = float(rewards_np.std(ddof=1)) if len(rewards_np) > 1 else 0.0

    # Histogram plot
    plt.figure(figsize=(6, 4))
    plt.hist(rewards_np, bins=30, alpha=0.75, edgecolor="k")
    plt.xlabel("Episode Reward")
    plt.ylabel("Count")
    plt.title(f"Reward Histogram ({args.eval_episodes} episodes)\nMean={mean_r:.1f}, Std={std_r:.1f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "plots", "eval_reward_histogram.png"), dpi=150)
    plt.close()

    stats = {"episodes": args.eval_episodes, "mean": mean_r, "std": std_r}
    with open(os.path.join(args.outdir, "logs", "eval_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f)

    print(json.dumps(stats, indent=2))


# -----------------------------
# CLI
# -----------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Deep Q-Network on MsPacman-v5 with image inputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Modes
    p.add_argument("--mode", choices=["train", "eval"], default="train", help="Run mode")

    # Training
    p.add_argument("--train-episodes", type=int, default=200, help="Number of training episodes")
    p.add_argument("--max-steps-per-episode", type=int, default=10000, help="Max steps per episode")
    p.add_argument("--replay-size", type=int, default=200_000, help="Replay buffer capacity")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--learning-rate", type=float, default=2.5e-4, help="Base learning rate")
    p.add_argument("--optimizer", choices=["adam", "rmsprop"], default="rmsprop", help="Optimizer choice")
    p.add_argument("--lr-gamma", type=float, default=0.999995, help="Exponential LR scheduler gamma per update step")
    p.add_argument("--epsilon-start", type=float, default=1.0, help="Starting epsilon for epsilon-greedy policy")
    p.add_argument("--epsilon-end", type=float, default=0.05, help="Final epsilon value")
    p.add_argument("--epsilon-decay-steps", type=int, default=1_000_000, help="Steps over which epsilon decays")
    p.add_argument("--learning-starts", type=int, default=20_000, help="Number of steps before learning starts")
    p.add_argument("--train-freq", type=int, default=4, help="How often to do a gradient update (in steps)")
    p.add_argument("--target-update-freq", type=int, default=10_000, help="Target network hard update frequency in steps")
    p.add_argument("--tau", type=float, default=0.0, help="Soft update factor; 0 = hard update only")
    p.add_argument("--frame-stack", type=int, default=4, help="Number of frames to stack")
    p.add_argument("--max-grad-norm", type=float, default=10.0, help="Gradient norm clipping")
    p.add_argument("--double-dqn", action="store_true", help="Use Double DQN targets")
    p.add_argument("--checkpoint-interval", type=int, default=50, help="Episodes between checkpoints")
    p.add_argument("--status-interval", type=int, default=1, help="Episode interval for status prints")

    # Enhancements
    p.add_argument("--dueling", action="store_true", help="Use Dueling DQN architecture")
    p.add_argument("--noisy-net", dest="noisy_net", action="store_true", help="Use NoisyLinear layers for exploration")
    p.add_argument("--hidden-dim", type=int, default=512, help="Hidden units in the MLP head")
    p.add_argument("--per", action="store_true", help="Use Prioritized Experience Replay")
    p.add_argument("--per-alpha", type=float, default=0.6, help="PER alpha (priority exponent)")
    p.add_argument("--per-beta0", type=float, default=0.4, help="Initial PER beta for importance sampling")
    p.add_argument("--per-beta-steps", type=int, default=1_000_000, help="Steps to anneal PER beta to 1.0")
    p.add_argument("--per-eps", type=float, default=1e-6, help="Priority epsilon to avoid zero priority")
    p.add_argument("--n-step", type=int, default=3, help="Use n-step returns for training targets")
    p.add_argument("--reward-clip", type=float, default=1.0, help="Clip per-step rewards to [-v, v] during training; set 0 to disable")
    p.add_argument("--terminal-on-life-loss", action="store_true", help="Treat life loss as terminal for training targets (Atari-specific)")

    # Common/eval
    p.add_argument("--eval-episodes", type=int, default=500, help="Number of evaluation episodes")
    p.add_argument("--model", type=str, default=None, help="Path to model checkpoint for eval")
    p.add_argument("--device", type=str, default=None, help="Device to use: 'cuda'|'cpu'|'mps'")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--outdir", type=str, default="runs/mspacman_dqn", help="Output directory for logs/plots/checkpoints")

    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
