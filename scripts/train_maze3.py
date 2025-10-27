import numpy as np
from tqdm import trange
import torch
from env.maze_env import MazeEnvironment
from agents.dqn_agent import DQNAgent
from utils.plot_utils import plot_rewards
from config import CONFIG
import os


def train_dqn():
    env_cfg = CONFIG["env"]
    dqn_cfg = CONFIG["dqn"]

    env = MazeEnvironment(**env_cfg)
    agent = DQNAgent(env,
                     lr=dqn_cfg["alpha"],
                     gamma=dqn_cfg["gamma"],
                     buffer_size=dqn_cfg["buffer_size"],
                     batch_size=dqn_cfg["batch_size"],
                     epsilon_start=dqn_cfg["epsilon_start"],
                     epsilon_end=dqn_cfg["epsilon_end"],
                     decay_episodes=500)

    episodes = 1000  # per spec  :contentReference[oaicite:4]{index=4}
    max_steps = 1000
    rewards = []

    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    for ep in trange(episodes):
        s = env.reset()
        total = 0

        # Target net handling (freeze every 10th)
        agent.maybe_sync_target(ep)

        for t in range(max_steps):
            a = agent.select_action(s, ep)
            ns, r, done = env.step(a)
            agent.remember(s, a, r, ns, done)
            agent.update_model()

            s = ns
            total += r
            if done:
                break

        rewards.append(total)
        agent.decay_epsilon(ep)

    np.save("data/rewards_maze3.npy", rewards)
    plot_rewards(rewards, "Milestone 3 – DQN Rewards", "plots/rewards_maze3.png")
    print("✅ DQN training complete. Saved rewards + plot.")

    # ----- Evaluation (100 episodes, greedy) -----
    successes, total_steps = 0, 0
    agent.epsilon = 0.0
    for _ in range(100):
        s = env.reset()
        for step in range(max_steps):
            with torch.no_grad():
                # greedy action from current policy
                from agents.dqn_agent import one_hot_state
                x = one_hot_state(s, env.N * env.M).to(agent.device)
                q = agent.model(x)
                a = int(torch.argmax(q).item())
            ns, r, done = env.step(a)
            s = ns
            if done:
                successes += 1
                total_steps += step + 1
                break

    avg_steps = (total_steps / successes) if successes > 0 else float('inf')
    print(f"Success Rate: {successes}%  |  Avg Steps: {avg_steps:.1f}")


if __name__ == "__main__":
    train_dqn()
