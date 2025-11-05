import os
from typing import Optional
import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym


class QNetwork(nn.Module):
    """Simple MLP Q-Network for low-dimensional observations."""
    def __init__(self, obs_dim: int, hidden_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors depending on stored type (np.ndarray or tensor)
        if isinstance(states[0], np.ndarray):
            states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        else:
            states = torch.stack(states).to(device)
            next_states = torch.stack(next_states).to(device)

        actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def train_dqn(
    env,
    device,
    obs_dim,
    act_dim,
    preprocess_obs=None,
    gamma=0.95,
    lr=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=5000,
    batch_size=64,
    buffer_size=5000,
    min_buffer=1000,
    target_update_freq=500,
    training_episodes=1000,
    hidden_dim=64,
    use_epsilon_scheduler=False,   
    use_huber_loss=False,
):
    """Train a DQN agent and return the trained network and logs."""

    policy_net = QNetwork(obs_dim, hidden_dim, act_dim).to(device)
    target_net = QNetwork(obs_dim, hidden_dim, act_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer(buffer_size)
    all_rewards = []
    max_q_values = []

    total_steps = 0

    def epsilon_by_step(step: int) -> float:
        return epsilon_end + (epsilon_start - epsilon_end) * np.exp(-step / epsilon_decay)

    for episode in range(training_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_max_q = []

        while not done:
            total_steps += 1

            # Preprocess observation if needed
            obs_t = torch.as_tensor(
                preprocess_obs(obs) if preprocess_obs else obs,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            # Epsilon-greedy exploration
            if use_epsilon_scheduler:
                epsilon = epsilon_by_step(total_steps)
            else:
                epsilon = epsilon_end

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(obs_t)
                    action = int(torch.argmax(q_values, dim=1).item())
                    episode_max_q.append(q_values.max().item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            if preprocess_obs:
                state_tensor = torch.as_tensor(preprocess_obs(obs), dtype=torch.float32, device=device)
                next_state_tensor = torch.as_tensor(preprocess_obs(next_obs), dtype=torch.float32, device=device)
                replay.push(state_tensor, action, reward, next_state_tensor, done)
            else:
                replay.push(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward

            # Optimize when enough samples are in buffer
            if len(replay) >= min_buffer:
                states, actions, rewards_b, next_states, dones = replay.sample(batch_size, device)

                with torch.no_grad():
                    next_q = target_net(next_states).max(1, keepdim=True)[0]
                    target_q = rewards_b + gamma * next_q * (1 - dones)

                q_values = policy_net(states).gather(1, actions)

                if use_huber_loss:
                    loss = nn.functional.smooth_l1_loss(q_values, target_q)
                else:
                    loss = nn.functional.mse_loss(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if total_steps % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        all_rewards.append(episode_reward)
        max_q_values.append(np.mean(episode_max_q) if episode_max_q else 0.0)

        if episode % 100 == 0:
            print(f"Episode {episode:4d} | Reward: {episode_reward:6.1f} | epsilon={epsilon:.2f}")

    env.close()
    return policy_net, all_rewards, max_q_values


def rollout_dqn(env, q_network: nn.Module, device: torch.device, episodes: int = 500, preprocess_obs=None, action_map=None):
    """Evaluate the trained Q-network greedily for a number of episodes."""
    q_network.eval()
    rewards = []

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            if preprocess_obs is not None:
                processed = preprocess_obs(obs)
                if isinstance(processed, np.ndarray):
                    obs_t = torch.as_tensor(processed, dtype=torch.float32, device=device).unsqueeze(0)
                else:
                    obs_t = processed.to(device).unsqueeze(0)
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                q_values = q_network(obs_t)
                action = int(torch.argmax(q_values, dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            obs = next_obs

        rewards.append(total_reward)

        if episode % 50 == 0:
            print(f"Eval Episode {episode:4d} | Return: {total_reward:.1f}")

    env.close()
    return np.array(rewards)


def plot_training_rewards(game_name: str, rewards, ma_window: int = 100, save_dir: Optional[str] = None, save_name: Optional[str] = None):
    plt.figure(figsize=(8, 5))
    plt.plot(rewards, label="Episode Reward", alpha=0.6)
    if len(rewards) >= ma_window:
        moving_avg = np.convolve(rewards, np.ones(ma_window) / ma_window, mode="valid")
        plt.plot(range(ma_window - 1, len(rewards)), moving_avg, label=f"{ma_window}-Episode Moving Average", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{game_name}: Training Rewards over Episodes")
    plt.legend()
    plt.grid(True)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = save_name or f"{game_name.lower()}_training_rewards.png"
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.show()
    plt.close()


def plot_rollout_histogram(game_name: str, rewards, save_dir: Optional[str] = None, save_name: Optional[str] = None):
    plt.figure(figsize=(7, 5))
    plt.hist(rewards, bins=20, color="steelblue", edgecolor="black", alpha=0.7)
    plt.title(f"{game_name}: Histogram of Rewards (500 Evaluation Episodes)")
    plt.xlabel("Episode Reward")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = save_name or f"{game_name.lower()}_rollout_histogram.png"
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.show()
    plt.close()

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"{game_name}: Mean reward over 500 episodes: {mean_reward:.2f}")
    print(f"{game_name}: Std. dev. of reward: {std_reward:.2f}")


def plot_max_q_values(game_name: str, max_q_values, save_dir: Optional[str] = None, save_name: Optional[str] = None):
    plt.figure(figsize=(8, 5))
    plt.plot(max_q_values, label="Average Max Q per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Max Q-Value")
    plt.title(f"{game_name}: Maximum Q-Values over Training")
    plt.legend()
    plt.grid(True)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = save_name or f"{game_name.lower()}_max_q_values.png"
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.show()
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Training environment
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Output directory for plots (relative to this script's folder)
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.join(script_dir, "plots")

    # Hyperparameters aligned with res_cartpole.ipynb
    trained_q_network, training_rewards, training_max_q_values = train_dqn(
        env=env,
        device=device,
        obs_dim=obs_dim,
        act_dim=act_dim,
        preprocess_obs=None,
        gamma=0.95,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=5000,
        batch_size=64,
        buffer_size=5000,
        min_buffer=1000,
        target_update_freq=500,
        training_episodes=2000,
        hidden_dim=64,
        use_epsilon_scheduler=False,
        use_huber_loss=False,
    )

    # Plots
    plot_max_q_values(game_name="CartPole", max_q_values=training_max_q_values,
                      save_dir=output_dir, save_name="cartpole_max_q_values.png")
    plot_training_rewards(game_name="CartPole", rewards=training_rewards,
                          save_dir=output_dir, save_name="cartpole_training_rewards.png")

    # Evaluation rollout (fresh env since train_dqn closes the original)
    eval_env = gym.make("CartPole-v1")
    rollout_rewards = rollout_dqn(eval_env, q_network=trained_q_network, device=device, episodes=500)
    plot_rollout_histogram(game_name="CartPole", rewards=rollout_rewards,
                           save_dir=output_dir, save_name="cartpole_rollout_histogram.png")


if __name__ == "__main__":
    main()
