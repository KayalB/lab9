import numpy as np
import pickle
import random
from agents.q_agent import QLearningAgent


class QLearningAgentWithHistory(QLearningAgent):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.5):
        super().__init__(env, alpha, gamma, epsilon)
        self.historical_data = None
        self.historical_states = set()

    def load_historical_data(self, path="data/historical_data.pkl"):
        try:
            with open(path, "rb") as f:
                self.historical_data = pickle.load(f)
            print(f"Loaded historical dataset with {len(self.historical_data)} transitions.")
            self.historical_states = {s for (s, a, r, ns) in self.historical_data}
        except FileNotFoundError:
            print("⚠️ No historical data found. Will train from scratch.")

    def pretrain_from_history(self):
        """Perform one Q-update per (s,a,r,s′) in the stored trajectories."""
        if self.historical_data is None:
            return
        for (s, a, r, ns) in self.historical_data:
            alpha_used = 0.15 if s in self.historical_states else self.alpha
            best_next = np.max(self.q_table[ns])
            self.q_table[s, a] += alpha_used * (r + self.gamma * best_next - self.q_table[s, a])

    def update(self, state, action, reward, next_state):
        """Override update to use higher learning rate for historical states."""
        # Use α = 0.15 for states that appeared in historical data, otherwise use normal α
        alpha_used = 0.15 if state in self.historical_states else self.alpha
        
        best_next = np.max(self.q_table[next_state])
        self.q_table[state, action] += alpha_used * (
            reward + self.gamma * best_next - self.q_table[state, action]
        )