import numpy as np
import random

class MazeEnvironment:
    def __init__(self, N, M, wall_prob=0.1):
        self.N, self.M = N, M
        self.start = (0, 0)
        self.goal = (N - 1, M - 1)
        self.grid = np.zeros((N, M))
        self.grid[self.goal] = 2
        self._add_walls(wall_prob)
        self.reset()

    def _add_walls(self, wall_prob):
        for i in range(self.N):
            for j in range(self.M):
                if (i, j) not in [self.start, self.goal] and random.random() < wall_prob:
                    self.grid[i][j] = 1  # wall

    def reset(self):
        self.agent_pos = list(self.start)
        return self._get_state_index()

    def _get_state_index(self):
        return self.agent_pos[0] * self.M + self.agent_pos[1]

    def step(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # Up, Down, Left, Right
        dx, dy = moves[action]
        new_x, new_y = self.agent_pos[0] + dx, self.agent_pos[1] + dy

        # Invalid move
        if not (0 <= new_x < self.N and 0 <= new_y < self.M) or self.grid[new_x][new_y] == 1:
            reward = -10
            return self._get_state_index(), reward, False

        self.agent_pos = [new_x, new_y]
        if (new_x, new_y) == self.goal:
            return self._get_state_index(), 100, True

        return self._get_state_index(), -1, False
