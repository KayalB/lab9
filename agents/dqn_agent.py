import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def one_hot_state(s_idx: int, size: int) -> torch.Tensor:
    x = torch.zeros(size, dtype=torch.float32)
    x[s_idx] = 1.0
    return x


class DQN(nn.Module):
    """
    Two hidden layers (64 each), ReLU, dropout(0.3) after first hidden layer,
    output size = 4 actions. (Spec matches Milestone 3.)  :contentReference[oaicite:2]{index=2}
    """
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)  # custom constraint
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class DQNAgent:
    """
    DQN with:
      - replay buffer size 1000
      - batch size 32
      - lr = 1e-3, gamma = 0.9
      - ε-greedy decays 0.9 → 0.1 over 500 episodes
      - Huber loss
      - target net freezing every 10th episode for 1 episode  :contentReference[oaicite:3]{index=3}
    """
    def __init__(self, env, lr=1e-3, gamma=0.9, buffer_size=1000, batch_size=32,
                 epsilon_start=0.9, epsilon_end=0.1, decay_episodes=500, device=None):
        self.env = env
        self.state_size = env.N * env.M
        self.action_size = 4

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(self.state_size, self.action_size).to(self.device)
        self.target_model = DQN(self.state_size, self.action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.HuberLoss()  # custom loss

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_episodes = decay_episodes

        self._freeze_target_this_episode = False  # for the “freeze every 10th” rule

    # ---------- policy / memory ----------
    def select_action(self, state_idx: int, episode: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            x = one_hot_state(state_idx, self.state_size).to(self.device)
            q = self.model(x)
            return int(torch.argmax(q).item())

    def remember(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))

    def _sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r, dtype=np.float32), np.array(ns), np.array(d, dtype=np.bool_)

    # ---------- training ----------
    def update_model(self):
        if len(self.memory) < self.batch_size:
            return  # not enough to train

        s, a, r, ns, d = self._sample_batch()

        # tensors
        S = torch.stack([one_hot_state(si, self.state_size) for si in s]).to(self.device)
        NS = torch.stack([one_hot_state(nsi, self.state_size) for nsi in ns]).to(self.device)
        A = torch.tensor(a, dtype=torch.int64, device=self.device)
        R = torch.tensor(r, dtype=torch.float32, device=self.device)
        D = torch.tensor(d, dtype=torch.bool, device=self.device)

        # current Q(s,a)
        q_values = self.model(S)
        q_sa = q_values.gather(1, A.view(-1, 1)).squeeze(1)

        # target: r + gamma * max_a' Q_target(ns, a') * (1 - done)
        with torch.no_grad():
            q_next = self.target_model(NS)
            max_q_next, _ = torch.max(q_next, dim=1)
            target = R + (~D).float() * self.gamma * max_q_next

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self, episode: int):
        # linear decay across decay_episodes
        frac = min(episode / max(1, self.decay_episodes), 1.0)
        self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac

    def maybe_sync_target(self, episode: int):
        # Custom rule: every 10th episode, freeze the target model *for that episode*.
        # We implement "freeze" by simply not syncing on that episode.
        is_freeze = (episode % 10 == 0)
        if is_freeze:
            self._freeze_target_this_episode = True
            return
        # on non-freeze episodes, if we were frozen last episode, unfreeze but still no special action needed
        if not self._freeze_target_this_episode:
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            # end of the freeze; next call will resume normal syncing
            self._freeze_target_this_episode = False
            self.target_model.load_state_dict(self.model.state_dict())
