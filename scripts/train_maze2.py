import numpy as np
import pickle
from tqdm import trange
from env.maze_env import MazeEnvironment
from agents.q_agent_hist import QLearningAgentWithHistory
from utils.plot_utils import plot_rewards, plot_path_heatmap
from config import CONFIG
import os

HIST_PATH = "data/historical_data.pkl"


def collect_historical_data(agent, env, num_episodes=500, max_steps=1000):
    """Run episodes with Milestone 1 agent and keep those that reach goal <200 steps."""
    print("🏁 Collecting historical trajectories…")
    good_transitions = []

    for ep in trange(num_episodes):
        s = env.reset()
        episode_transitions = []
        for step in range(max_steps):
            a = agent.choose_action(s, ep)
            ns, r, done = env.step(a)
            episode_transitions.append((s, a, r, ns))
            agent.update(s, a, r, ns)
            s = ns
            if done:
                if step < 200:
                    good_transitions.extend(episode_transitions)
                break

    os.makedirs("data", exist_ok=True)
    with open(HIST_PATH, "wb") as f:
        pickle.dump(good_transitions, f)
    print(f"✅ Saved {len(good_transitions)} total transitions to {HIST_PATH}")


def train_with_history():
    env_cfg = CONFIG["env"]
    q_cfg = CONFIG["q_learning"]
    env = MazeEnvironment(**env_cfg)
    agent = QLearningAgentWithHistory(env,
                                      alpha=q_cfg["alpha"],
                                      gamma=q_cfg["gamma"],
                                      epsilon=0.5)

    # 1️⃣ Generate historical data if not already present
    if not os.path.exists(HIST_PATH):
        collect_historical_data(agent, env)
    agent.load_historical_data(HIST_PATH)

    # 2️⃣ Pre-train from dataset
    agent.pretrain_from_history()

    # 3️⃣ Continue training (500 episodes)
    print("🎯 Continuing training with historical initialization…")
    rewards = []
    training_paths = []  # Collect paths during training for heatmap
    
    for ep in trange(500):
        s = env.reset()
        total = 0
        episode_path = [tuple(env.agent_pos)]  # Track path for this episode
        
        for step in range(q_cfg["max_steps"]):
            a = agent.choose_action(s, ep)
            ns, r, done = env.step(a)
            agent.update(s, a, r, ns)
            episode_path.append(tuple(env.agent_pos))
            s = ns
            total += r
            if done:
                # Collect paths periodically during training (every 5 episodes)
                if ep % 5 == 0:
                    training_paths.append(episode_path)
                break
        
        agent.epsilon = max(q_cfg["epsilon_end"],
                            agent.epsilon - 0.8 / 500)
        rewards.append(total)

    np.save("data/rewards_maze2.npy", rewards)
    plot_rewards(rewards,
                 "Milestone 2 – Q-Learning w/ Historical Data",
                 "plots/rewards_maze2.png")
    print("✅ Training complete. Plots + data saved.")

    # 4️⃣ Simple evaluation
    successes, total_steps = 0, 0
    
    for episode_num in range(100):
        s = env.reset()
        
        for step in range(q_cfg["max_steps"]):
            a = np.argmax(agent.q_table[s])
            ns, r, done = env.step(a)
            s = ns
            if done:
                successes += 1
                total_steps += step + 1
                break

    print(f"Success Rate = {successes}%")
    print(f"Average Steps = {total_steps / max(successes,1):.1f}")
    print("➡️ Compare this with Milestone 1 results in your report.")
    
    # Generate heatmap visualization from training exploration
    if training_paths:
        print("\n" + "="*50)
        print("Generating training exploration heatmap...")
        print("="*50)
        plot_path_heatmap(
            env, 
            training_paths, 
            "Maze 2 - Training Exploration Heatmap (w/ Historical Data)",
            "plots/path_maze2.png"
        )
    else:
        print("\nNo successful training paths found to visualize.")


if __name__ == "__main__":
    train_with_history()
