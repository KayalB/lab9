# Lab 9: Reinforcement Learning in Maze Environments

This project implements and compares three different reinforcement learning approaches for solving maze navigation tasks.

## Project Structure

```
lab9/
├── agents/                 # RL agent implementations
│   ├── q_agent.py         # Standard Q-Learning agent
│   ├── q_agent_hist.py    # Q-Learning with historical data
│   └── dqn_agent.py       # Deep Q-Network agent
├── env/                    # Environment implementation
│   └── maze_env.py        # Maze environment
├── scripts/               # Training scripts
│   ├── train_maze1.py     # Milestone 1: Standard Q-Learning
│   ├── train_maze2.py     # Milestone 2: Q-Learning with history
│   └── train_maze3.py     # Milestone 3: DQN
├── utils/                 # Utility functions
│   └── plot_utils.py      # Plotting utilities
├── data/                  # Generated data files
├── plots/                 # Generated plots
├── config.py              # Configuration parameters
├── run_all_mazes.py       # Run all experiments and compare
└── requirements.txt       # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run All Experiments (Recommended)

To run all three maze experiments and generate a performance comparison:

```bash
python run_all_mazes.py
```

This will:
- Train all three models (Maze 1, 2, and 3)
- Evaluate each model with 100 test episodes
- Generate reward plots for each milestone
- Create a comprehensive performance summary in `data/performance_summary.txt`

### Run Individual Experiments

Run each milestone separately:

```bash
# Milestone 1: Standard Q-Learning with custom exploration
python scripts/train_maze1.py

# Milestone 2: Q-Learning with historical data pretraining
python scripts/train_maze2.py

# Milestone 3: Deep Q-Network (DQN)
python scripts/train_maze3.py
```

## Implementations

### Milestone 1: Q-Learning with Custom Exploration
- Standard Q-Learning algorithm
- Custom exploration: Every 5th episode performs 3 random actions
- 1000 training episodes
- Epsilon decay from 0.9 to 0.1

### Milestone 2: Q-Learning with Historical Data
- Uses historical trajectories for initialization
- Collects 500 episodes of good trajectories (< 200 steps)
- Pretrains Q-table from historical data
- 500 additional training episodes

### Milestone 3: Deep Q-Network (DQN)
- Neural network-based Q-learning
- Experience replay buffer (size 1000)
- Target network with periodic freezing
- Huber loss function
- 1000 training episodes

## Evaluation

All models are evaluated with:
- 100 test episodes
- Greedy action selection (epsilon = 0)
- Metrics tracked: success rate, average steps to goal

## Configuration

Edit `config.py` to modify:
- Maze size (N x M)
- Wall probability
- Learning rate (alpha)
- Discount factor (gamma)
- Training episodes
- Epsilon decay parameters

## Output Files

After running experiments, the following files are generated:

- `data/rewards_maze1.npy` - Training rewards for Maze 1
- `data/rewards_maze2.npy` - Training rewards for Maze 2
- `data/rewards_maze3.npy` - Training rewards for Maze 3
- `data/historical_data.pkl` - Historical trajectories (Maze 2)
- `data/performance_summary.txt` - Comprehensive comparison report
- `plots/rewards_maze1.png` - Reward plot for Maze 1
- `plots/rewards_maze2.png` - Reward plot for Maze 2
- `plots/rewards_maze3.png` - Reward plot for Maze 3

## Requirements

- Python 3.8+
- NumPy
- PyTorch
- Matplotlib
- tqdm
