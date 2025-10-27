import numpy as np
from env.maze_env import MazeEnvironment
from agents.q_agent import QLearningAgent
from utils.plot_utils import plot_rewards
from config import CONFIG
from tqdm import trange

def train():
    env_cfg = CONFIG["env"]
    q_cfg = CONFIG["q_learning"]

    env = MazeEnvironment(**env_cfg)
    agent = QLearningAgent(env, q_cfg["alpha"], q_cfg["gamma"], q_cfg["epsilon_start"])

    rewards = []
    for episode in trange(q_cfg["episodes"]):
        state = env.reset()
        total_reward = 0

        for step in range(q_cfg["max_steps"]):
            action = agent.choose_action(state, episode)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.epsilon = max(q_cfg["epsilon_end"],
                            q_cfg["epsilon_start"] - episode / q_cfg["episodes"] * 0.8)
        rewards.append(total_reward)

    np.save("data/rewards_maze1.npy", rewards)
    plot_rewards(rewards, "Milestone 1 - Q-Learning Rewards", "plots/rewards_maze1.png")
    print("Training complete. Results saved in /data and /plots.")

if __name__ == "__main__":
    train()
