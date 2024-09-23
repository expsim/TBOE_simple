import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


class MultiAgentDQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(MultiAgentDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)

        # Separate output layers for each agent
        self.agent_heads = nn.ModuleList([nn.Linear(128, action_dim) for _ in range(4)])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Return Q-values for each agent
        return [head(x) for head in self.agent_heads]

class Octopus:
    def __init__(self, state_dim, action_dim, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay_steps=10000):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = MultiAgentDQN(state_dim, action_dim).to(self.device)
        self.target_network = MultiAgentDQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.memory = []
        self.max_memory_size = 100000

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return [np.random.randint(self.action_dim) for _ in range(4)]

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return [q.argmax().item() for q in q_values]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)

        total_loss = 0
        for i in range(4):  # For each agent
            current_q = current_q_values[i].gather(1, actions[:, i].unsqueeze(1)).squeeze(1)
            next_q_max = next_q_values[i].max(1)[0].detach()
            expected_q = rewards + self.gamma * next_q_max * (1 - dones)
            loss = F.smooth_l1_loss(current_q, expected_q)  # Huber loss for stability
            total_loss += loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1)  # Gradient clipping
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())