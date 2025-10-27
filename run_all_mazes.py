#!/usr/bin/env python3
"""
Run all maze training scripts and generate a performance comparison summary.
"""
import numpy as np
import time
from datetime import datetime
from env.maze_env import MazeEnvironment
from agents.q_agent import QLearningAgent
from agents.q_agent_hist import QLearningAgentWithHistory
from agents.dqn_agent import DQNAgent
from utils.plot_utils import plot_rewards
from config import CONFIG
from tqdm import trange
import os
import pickle
import torch

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)


def evaluate_agent(agent, env, max_steps, num_episodes=100, agent_type="q_learning"):
    """
    Evaluate an agent's performance with greedy action selection.
    
    Returns:
        dict: Dictionary with success_rate, avg_steps, total_steps, successes
    """
    successes, total_steps = 0, 0
    step_counts = []
    
    for _ in range(num_episodes):
        s = env.reset()
        for step in range(max_steps):
            if agent_type == "dqn":
                with torch.no_grad():
                    from agents.dqn_agent import one_hot_state
                    x = one_hot_state(s, env.N * env.M).to(agent.device)
                    q = agent.model(x)
                    a = int(torch.argmax(q).item())
            else:  # q_learning
                a = np.argmax(agent.q_table[s])
            
            ns, r, done = env.step(a)
            s = ns
            if done:
                successes += 1
                steps_taken = step + 1
                total_steps += steps_taken
                step_counts.append(steps_taken)
                break
        else:
            step_counts.append(-1)  # Mark as failure
    
    avg_steps = total_steps / max(successes, 1)
    
    return {
        "success_rate": successes,
        "avg_steps": avg_steps,
        "total_steps": total_steps,
        "successes": successes,
        "step_counts": step_counts,
        "min_steps": min([s for s in step_counts if s > 0], default=0),
        "max_steps": max([s for s in step_counts if s > 0], default=0),
        "median_steps": np.median([s for s in step_counts if s > 0]) if any(s > 0 for s in step_counts) else 0
    }


def train_maze1():
    """Milestone 1: Standard Q-Learning with custom exploration"""
    print("\n" + "="*70)
    print("MILESTONE 1: Q-Learning with Custom Exploration")
    print("="*70)
    
    start_time = time.time()
    env_cfg = CONFIG["env"]
    q_cfg = CONFIG["q_learning"]

    env = MazeEnvironment(**env_cfg)
    agent = QLearningAgent(env, q_cfg["alpha"], q_cfg["gamma"], q_cfg["epsilon_start"])

    rewards = []
    for episode in trange(q_cfg["episodes"], desc="Training Maze 1"):
        state = env.reset()
        total_reward = 0

        done = False
        # Custom exploration: every 5th episode → 3 consecutive random actions
        if (episode + 1) % 5 == 0:
            for _ in range(3):
                action = np.random.randint(0, 4)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                if done:
                    break
        
        # Continue with normal epsilon-greedy
        if not done:
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
    
    training_time = time.time() - start_time
    
    # Evaluate
    print("\nEvaluating Maze 1...")
    results = evaluate_agent(agent, env, q_cfg["max_steps"], agent_type="q_learning")
    results["training_time"] = training_time
    results["total_episodes"] = q_cfg["episodes"]
    results["avg_reward"] = np.mean(rewards[-100:])  # Last 100 episodes
    
    print(f"Maze 1 completed - Success Rate: {results['success_rate']}%, Avg Steps: {results['avg_steps']:.1f}")
    return results, env


def train_maze2():
    """Milestone 2: Q-Learning with Historical Data"""
    print("\n" + "="*70)
    print("MILESTONE 2: Q-Learning with Historical Data")
    print("="*70)
    
    start_time = time.time()
    HIST_PATH = "data/historical_data.pkl"
    env_cfg = CONFIG["env"]
    q_cfg = CONFIG["q_learning"]
    env = MazeEnvironment(**env_cfg)
    
    # Track total episodes including historical data collection
    total_training_episodes = 0
    history_episodes = 500
    
    agent = QLearningAgentWithHistory(env,
                                      alpha=q_cfg["alpha"],
                                      gamma=q_cfg["gamma"],
                                      epsilon=0.5)

    # Generate historical data if needed
    if not os.path.exists(HIST_PATH):
        print("Collecting historical trajectories…")
        good_transitions = []
        for ep in trange(history_episodes, desc="Collecting History"):
            s = env.reset()
            episode_transitions = []
            for step in range(1000):
                a = agent.choose_action(s, ep)
                ns, r, done = env.step(a)
                episode_transitions.append((s, a, r, ns))
                agent.update(s, a, r, ns)
                s = ns
                if done:
                    if step < 200:
                        good_transitions.extend(episode_transitions)
                    break
        
        total_training_episodes += history_episodes
        with open(HIST_PATH, "wb") as f:
            pickle.dump(good_transitions, f)
        print(f"Saved {len(good_transitions)} transitions to {HIST_PATH}")
    else:
        # Historical data already exists, count those episodes too
        total_training_episodes += history_episodes
    
    agent.load_historical_data(HIST_PATH)
    agent.pretrain_from_history()

    # Continue training
    print("Continuing training with historical initialization…")
    rewards = []
    continued_episodes = 500
    for ep in trange(continued_episodes, desc="Training Maze 2"):
        s = env.reset()
        total = 0
        for step in range(q_cfg["max_steps"]):
            a = agent.choose_action(s, ep)
            ns, r, done = env.step(a)
            agent.update(s, a, r, ns)
            s = ns
            total += r
            if done:
                break
        agent.epsilon = max(q_cfg["epsilon_end"], agent.epsilon - 0.8 / continued_episodes)
        rewards.append(total)
    
    total_training_episodes += continued_episodes

    np.save("data/rewards_maze2.npy", rewards)
    plot_rewards(rewards, "Milestone 2 – Q-Learning w/ Historical Data", "plots/rewards_maze2.png")
    
    training_time = time.time() - start_time
    
    # Evaluate
    print("\nEvaluating Maze 2...")
    results = evaluate_agent(agent, env, q_cfg["max_steps"], agent_type="q_learning")
    results["training_time"] = training_time
    results["total_episodes"] = total_training_episodes  # Total includes history collection (500) + continued training (500) = 1000
    results["avg_reward"] = np.mean(rewards[-100:])
    
    print(f"Maze 2 completed - Success Rate: {results['success_rate']}%, Avg Steps: {results['avg_steps']:.1f}")
    return results, env


def train_maze3():
    """Milestone 3: Deep Q-Network (DQN)"""
    print("\n" + "="*70)
    print("MILESTONE 3: Deep Q-Network (DQN)")
    print("="*70)
    
    start_time = time.time()
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

    episodes = 1000
    max_steps = 1000
    rewards = []

    for ep in trange(episodes, desc="Training Maze 3"):
        s = env.reset()
        total = 0

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
    
    training_time = time.time() - start_time
    
    # Evaluate
    print("\nEvaluating Maze 3...")
    agent.epsilon = 0.0
    results = evaluate_agent(agent, env, max_steps, agent_type="dqn")
    results["training_time"] = training_time
    results["total_episodes"] = episodes
    results["avg_reward"] = np.mean(rewards[-100:])
    
    print(f"Maze 3 completed - Success Rate: {results['success_rate']}%, Avg Steps: {results['avg_steps']:.1f}")
    return results, env


def generate_summary(results_dict, env):
    """Generate a comprehensive performance summary"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary = f"""
{'='*80}
                    MAZE LEARNING - PERFORMANCE COMPARISON
{'='*80}
Generated: {timestamp}
Maze Configuration: {env.N}x{env.M} grid, {CONFIG['env']['wall_prob']*100}% wall probability
Minimum Possible Steps: {env.N + env.M - 2}
{'='*80}

"""
    
    # Table header
    summary += f"""
{'Metric':<35} {'Maze 1':<15} {'Maze 2':<15} {'Maze 3':<15}
{'-'*80}
"""
    
    # Add metrics
    metrics = [
        ("Method", "Q-Learning", "Q-Learning+Hist", "DQN"),
        ("Training Episodes", 
         f"{results_dict['maze1']['total_episodes']}", 
         f"{results_dict['maze2']['total_episodes']}", 
         f"{results_dict['maze3']['total_episodes']}"),
        ("Training Time (seconds)", 
         f"{results_dict['maze1']['training_time']:.2f}", 
         f"{results_dict['maze2']['training_time']:.2f}", 
         f"{results_dict['maze3']['training_time']:.2f}"),
        ("", "", "", ""),
        ("=== EVALUATION RESULTS (100 test episodes) ===", "", "", ""),
        ("Success Rate (%)", 
         f"{results_dict['maze1']['success_rate']}", 
         f"{results_dict['maze2']['success_rate']}", 
         f"{results_dict['maze3']['success_rate']}"),
        ("Avg Steps to Goal", 
         f"{results_dict['maze1']['avg_steps']:.2f}", 
         f"{results_dict['maze2']['avg_steps']:.2f}", 
         f"{results_dict['maze3']['avg_steps']:.2f}"),
        ("Min Steps", 
         f"{results_dict['maze1']['min_steps']}", 
         f"{results_dict['maze2']['min_steps']}", 
         f"{results_dict['maze3']['min_steps']}"),
        ("Max Steps", 
         f"{results_dict['maze1']['max_steps']}", 
         f"{results_dict['maze2']['max_steps']}", 
         f"{results_dict['maze3']['max_steps']}"),
        ("Median Steps", 
         f"{results_dict['maze1']['median_steps']:.2f}", 
         f"{results_dict['maze2']['median_steps']:.2f}", 
         f"{results_dict['maze3']['median_steps']:.2f}"),
        ("", "", "", ""),
        ("Avg Reward (last 100 episodes)", 
         f"{results_dict['maze1']['avg_reward']:.2f}", 
         f"{results_dict['maze2']['avg_reward']:.2f}", 
         f"{results_dict['maze3']['avg_reward']:.2f}"),
    ]
    
    for metric in metrics:
        if len(metric) == 4:
            summary += f"{metric[0]:<35} {metric[1]:<15} {metric[2]:<15} {metric[3]:<15}\n"
        else:
            summary += f"{metric[0]}\n"
    
    summary += f"\n{'='*80}\n"
    
    # Analysis section
    summary += """
ANALYSIS & INSIGHTS:
"""
    
    # Determine best performer
    best_success = max(results_dict['maze1']['success_rate'], 
                       results_dict['maze2']['success_rate'], 
                       results_dict['maze3']['success_rate'])
    
    best_steps = min(results_dict['maze1']['avg_steps'], 
                     results_dict['maze2']['avg_steps'], 
                     results_dict['maze3']['avg_steps'])
    
    summary += f"\n• Highest Success Rate: {best_success}%"
    for maze, results in results_dict.items():
        if results['success_rate'] == best_success:
            summary += f" ({maze.replace('_', ' ').title()})"
    
    summary += f"\n• Lowest Avg Steps: {best_steps:.2f}"
    for maze, results in results_dict.items():
        if abs(results['avg_steps'] - best_steps) < 0.01:
            summary += f" ({maze.replace('_', ' ').title()})"
    
    summary += "\n\n• Training Efficiency:"
    fastest = min(results_dict.values(), key=lambda x: x['training_time'])
    for maze, results in results_dict.items():
        if results['training_time'] == fastest['training_time']:
            summary += f"\n  - Fastest training: {maze.replace('_', ' ').title()} ({results['training_time']:.2f}s)"
    
    summary += "\n\n• Method Comparison:"
    summary += "\n  - Maze 1 (Standard Q-Learning): Baseline with custom exploration strategy"
    summary += "\n  - Maze 2 (Historical Data): Leverages pre-collected trajectories for faster learning"
    summary += "\n  - Maze 3 (DQN): Neural network-based approach with experience replay"
    
    summary += f"\n\n{'='*80}\n"
    summary += "Files Generated:\n"
    summary += "  • data/rewards_maze1.npy, data/rewards_maze2.npy, data/rewards_maze3.npy\n"
    summary += "  • plots/rewards_maze1.png, plots/rewards_maze2.png, plots/rewards_maze3.png\n"
    summary += "  • data/performance_summary.txt (this file)\n"
    summary += f"{'='*80}\n"
    
    return summary


def main():
    """Run all maze training scripts and generate comparison"""
    print("          RUNNING ALL MAZE EXPERIMENTS")
    
    overall_start = time.time()
    
    # Run all three maze experiments
    results1, env1 = train_maze1()
    results2, env2 = train_maze2()
    results3, env3 = train_maze3()
    
    overall_time = time.time() - overall_start
    
    # Compile results
    results_dict = {
        "maze1": results1,
        "maze2": results2,
        "maze3": results3
    }
    
    # Generate summary
    summary = generate_summary(results_dict, env1)
    
    # Save summary to file
    summary_path = "data/performance_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    
    # Print summary
    print(summary)
    
    print(f"\nTotal Execution Time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    print(f"Performance summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

