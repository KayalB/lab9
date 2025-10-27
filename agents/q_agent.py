import numpy as np
import random

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.9):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.random.uniform(-0.1, 0.1, (env.N * env.M, 4))

    def choose_action(self, state, episode):
        # Custom exploration: every 5th episode → 3 random actions
        if episode % 5 == 0 and random.random() < 0.5:
            return random.randint(0, 3)

        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (
            reward + self.gamma * best_next - self.q_table[state, action]
        )
