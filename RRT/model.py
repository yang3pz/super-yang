

import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim



class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256): 
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_logits = self.action_head(x)
        return torch.softmax(action_logits, dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256): 
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        state_values = self.value_head(x)
        return state_values


class PPOAgent:
    def __init__(self, state_dim, action_dim, device, hidden_dim=256, lr_policy=1e-4, lr_value=1e-3):
        self.device = device
        self.action_dim = action_dim

        # 初始化策略网络和价值网络
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.value_net = ValueNetwork(state_dim, hidden_dim).to(device)

        # 优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)

    def select_action(self, state):
        """根据当前策略选择动作，并返回动作、概率和价值估计"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.policy_net(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value_net(state)
        return action.item(), log_prob.item(), value.item()

    def compute_advantages(self, rewards, dones, values, next_values, gamma=0.99, gae_lambda=0.95):
        """计算优势函数和回报"""
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * next_values[step] * (1 - dones[step]) - values[step]
            gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values)]
        # 规范化优势
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages.tolist(), returns

    def update(self, trajectories, batch_size=64, epochs=10, clip_epsilon=0.1, entropy_coef=0.05, value_loss_coef=0.5, max_grad_norm=0.5):
        """执行PPO更新"""
        states = torch.FloatTensor(np.array([t['state'] for t in trajectories])).to(self.device)
        actions = torch.LongTensor(np.array([t['action'] for t in trajectories])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array([t['log_prob'] for t in trajectories])).to(self.device)
        advantages = torch.FloatTensor(np.array([t['advantage'] for t in trajectories])).to(self.device)
        returns = torch.FloatTensor(np.array([t['return'] for t in trajectories])).to(self.device)

        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, advantages, returns)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in dataloader:
                # 计算新的动作概率
                action_probs = self.policy_net(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # 计算比率
                ratios = torch.exp(new_log_probs - batch_old_log_probs)

                # 计算PPO的目标函数
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 计算价值网络的损失
                value_pred = self.value_net(batch_states).squeeze()
                value_loss = nn.MSELoss()(value_pred, batch_returns)

                # 总损失
                loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

                # 反向传播
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()
