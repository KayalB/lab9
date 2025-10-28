import numpy as np
from env.maze_env import MazeEnvironment
from agents.q_agent import QLearningAgent
from utils.plot_utils import plot_rewards, plot_path_heatmap
from config import CONFIG
from tqdm import trange

def train():
    env_cfg = CONFIG["env"]
    q_cfg = CONFIG["q_learning"]

    env = MazeEnvironment(**env_cfg)
    agent = QLearningAgent(env, q_cfg["alpha"], q_cfg["gamma"], q_cfg["epsilon_start"])

    rewards = []
    training_paths = []  # Collect paths during training for heatmap
    
    for episode in trange(q_cfg["episodes"]):
        state = env.reset()
        total_reward = 0
        episode_path = [tuple(env.agent_pos)]  # Track path for this episode

        done = False
        # Custom exploration: every 5th episode → 3 consecutive random actions
        if (episode + 1) % 5 == 0:
            for _ in range(3):
                action = np.random.randint(0, 4)  # Random action (0-3, not 0-2!)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state)
                episode_path.append(tuple(env.agent_pos))
                state = next_state
                total_reward += reward
                if done:
                    break
        
        # Continue with normal epsilon-greedy for the rest of the episode
        if not done:  # only continue if goal not reached during forced exploration
            for step in range(q_cfg["max_steps"]):
                action = agent.choose_action(state, episode)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state)
                episode_path.append(tuple(env.agent_pos))
                state = next_state
                total_reward += reward
                if done:
                    break

        # Collect paths periodically during training (every 10 episodes)
        if episode % 10 == 0 and done:
            training_paths.append(episode_path)

        agent.epsilon = max(q_cfg["epsilon_end"],
                            q_cfg["epsilon_start"] - episode / q_cfg["episodes"] * 0.8)
        rewards.append(total_reward)

    np.save("data/rewards_maze1.npy", rewards)
    plot_rewards(rewards, "Milestone 1 - Q-Learning Rewards", "plots/rewards_maze1.png")
    print("Training complete. Results saved in /data and /plots.")

    # Evaluation: 100 test episodes with greedy action selection (epsilon=0)
    print("\n" + "="*50)
    print("Evaluating agent performance...")
    print("="*50)
    successes, total_steps = 0, 0
    
    for episode_num in range(100):
        s = env.reset()
        
        for step in range(q_cfg["max_steps"]):
            a = np.argmax(agent.q_table[s])  # Greedy action selection
            ns, r, done = env.step(a)
            s = ns
            if done:
                successes += 1
                total_steps += step + 1
                break

    success_rate = successes
    avg_steps = total_steps / max(successes, 1)
    
    print(f"\n Evaluation results (100 test episodes):")
    print(f"   Success Rate: {success_rate}%")
    print(f"   Average Steps to Goal: {avg_steps:.1f}")
    print("="*50)
    print("Training maze walls:")
    print(env.grid)
    
    # Generate heatmap visualization from training exploration
    if training_paths:
        print("\n" + "="*50)
        print("Generating training exploration heatmap...")
        print("="*50)
        plot_path_heatmap(
            env, 
            training_paths, 
            "Maze 1 - Training Exploration Heatmap",
            "plots/path_maze1.png"
        )
    else:
        print("\nNo successful training paths found to visualize.")

if __name__ == "__main__":
    train()
