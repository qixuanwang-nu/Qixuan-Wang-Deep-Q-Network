**DQN: CartPole and MsPacman**

- Implements Deep Q-Network (DQN) for CartPole and MsPacman.
- Includes a self-contained CartPole training/evaluation script that saves plots.
- Provides evaluation summaries for both environments.

**Repository Layout**

- `DeepQNetwork/run_cartpole_dqn.py`: Self-contained CartPole DQN train + 500-episode eval + plots.
- `DeepQNetwork/train.py`: Generic DQN trainer used by notebooks/examples.
- `DeepQNetwork/qnetwork.py`: MLP Q-network for low-dimensional states.
- `DeepQNetwork/relaybuffer.py`: Replay buffer utilities.
- `DeepQNetwork/vis.py`: Plot helpers for rewards and Q-values.
- `DeepQNetwork/utils.py`: Misc utilities (e.g., Atari preprocessing helper).
- `train_mspacman_dqn.py`: MsPacman DQN training/evaluation entry point.
- `requirements.txt`: Project dependencies.

**Setup**

- Python 3.9+ recommended.
- Install: `pip install -r requirements.txt`
- For Atari (MsPacman): `pip install "gymnasium[atari,accept-rom-license]"`

**CartPole: Train, Evaluate, Plot**

- Run: `python DeepQNetwork/run_cartpole_dqn.py`
- Saves plots to: `DeepQNetwork/plots/`
- Files: `cartpole_training_rewards.png`, `cartpole_max_q_values.png`, `cartpole_rollout_histogram.png`
- Evaluation (sample log lines):
  - Eval Episode 0 | Return: 500.0
  - Eval Episode 50 | Return: 500.0
  - Eval Episode 100 | Return: 500.0
  - Eval Episode 150 | Return: 500.0
  - Eval Episode 200 | Return: 500.0
  - Eval Episode 250 | Return: 500.0
  - Eval Episode 300 | Return: 500.0
  - Eval Episode 350 | Return: 500.0
  - Eval Episode 400 | Return: 500.0
  - Eval Episode 450 | Return: 500.0
- Summary: consistent perfect returns; mean approximately 500.0 over evaluation episodes.

**MsPacman: Evaluation Summary**

- Stats file: `mspacman_logs/eval_stats.json`
- Parsed summary (500 episodes): mean = 2248.84, std = 83.11
- To re-run training/eval, use `train_mspacman_dqn.py` (outputs logs/plots under a chosen run directory).

**Notes**

- CartPole hyperparameters match `DeepQNetwork/res_cartpole.ipynb` (gamma=0.95, batch=64, target_update=500, constant epsilon=0.05 during training).
- The CartPole script closes the training env and creates a fresh env for evaluation to ensure clean rollouts.
