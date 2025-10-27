import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards, title, save_path):
    plt.figure()
    plt.plot(np.arange(len(rewards)), rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
