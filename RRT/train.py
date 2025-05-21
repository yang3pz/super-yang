
from env import RLenv

import numpy as np

import matplotlib.pyplot as plt
from model import PPOAgent

import torch
from utils import set_seed  


SEED = 42
set_seed(SEED)


def main(render=False):
  
    env = RLenv(map_file='map.csv', wall_distance=2, local_view=2, max_steps=1000,alpha=1.0)
    env.seed(SEED)  
    env.action_space.seed(SEED)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

   
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim, device)

    # 训练参数
    max_episodes = 300  
    update_timesteps = 2048
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.1  # 剪切范围
    entropy_coef = 0.05  # 熵系数
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    batch_size = 64
    epochs = 10

    
    episode_rewards = []
    average_rewards = []

    timestep = 0
    episode = 0

    while episode < max_episodes:
        trajectories = []
        collected_timesteps = 0
        while collected_timesteps < update_timesteps:
            state = env.reset()
            done = False
            ep_reward = 0
            while not done:
             
                if render and episode % 100 == 0 and episode > 0:
                    env.render()

                action, log_prob, value = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward

                trajectories.append({
                    'state': state,
                    'action': action,
                    'log_prob': log_prob,
                    'reward': reward,
                    'done': done,
                    'value': value
                })

                state = next_state
                timestep += 1
                collected_timesteps += 1

                if done:
                    episode += 1
                    episode_rewards.append(ep_reward)
                    if episode % 10 == 0:
                        avg_reward = np.mean(episode_rewards[-10:])
                        average_rewards.append(avg_reward)
                        print(f"回合 {episode}: 最近10回合的平均奖励 = {avg_reward:.2f}")

                if collected_timesteps >= update_timesteps:
                    break

        # 计算优势和回报
        rewards = [t['reward'] for t in trajectories]
        dones = [t['done'] for t in trajectories]
        values = [t['value'] for t in trajectories]
        next_values = values[1:] + [0]

        advantages, returns = agent.compute_advantages(rewards, dones, values, next_values, gamma, gae_lambda)

        # 更新轨迹中的优势和回报
        for i in range(len(trajectories)):
            trajectories[i]['advantage'] = advantages[i]
            trajectories[i]['return'] = returns[i]

        # 更新PPOAgent
        agent.update(trajectories, batch_size=batch_size, epochs=epochs, clip_epsilon=clip_epsilon,
                    entropy_coef=entropy_coef, value_loss_coef=value_loss_coef, max_grad_norm=max_grad_norm)

    
        trajectories = []


    torch.save(agent.policy_net.state_dict(), "ppo_policy_net.pth")
    torch.save(agent.value_net.state_dict(), "ppo_value_net.pth")
    print("训练完成，模型已保存。")


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.xlabel('回合')
    plt.ylabel('总奖励')
    plt.title('每回合总奖励')

    plt.subplot(1, 2, 2)
    plt.plot(average_rewards)
    plt.xlabel('每10回合')
    plt.ylabel('平均奖励')
    plt.title('最近10回合的平均奖励')

    plt.show()


if __name__ == "__main__":
  
    main(render=False)  
