import matplotlib.pyplot as plt
import torch
from utils import set_seed  
from env import RLenv
from model import PolicyNetwork
import matplotlib.patches as mpatches
from matplotlib import colors
import time

def test_agent(env, policy_net, device, num_episodes=10, render=False, seed=42, pause_time=0.1):
    """
    测试智能体并动态可视化其路径。

    参数:
    - env: 环境实例。
    - policy_net: 策略网络。
    - device: 设备（CPU或GPU）。
    - num_episodes: 测试的回合数。
    - render: 是否动态渲染。
    - seed: 随机种子。
    - pause_time: 动态更新时的暂停时间（秒）。
    """
    policy_net.eval()  
    success_count = 0
    total_steps = 0

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        path = [env.current_pos]
        total_reward = 0
        steps = 0

        if render:
            plt.ion()  # 启用交互模式
            fig, ax = plt.subplots(figsize=(6, 6))
            cmap = colors.ListedColormap(['white', 'black', 'red', 'blue'])
            bounds = [0, 1, 2, 3, 4]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            ax.imshow(env.map, cmap=cmap, norm=norm, origin='upper')

            # 绘制目标点
            for i, point in enumerate(env.waypoints):
                if i == env.current_waypoint_idx:
                    ax.plot(point[1], point[0], 'r*', markersize=10, label='当前目标')
                else:
                    if i == 0:
                        ax.plot(point[1], point[0], 'g*', markersize=10, label='目标点')
                    else:
                        ax.plot(point[1], point[0], 'g*', markersize=10)

            # 初始化路径线
            path_line, = ax.plot([], [], 'b-', linewidth=2, label='智能体路径')

            # 初始化起点和终点标记
            start_marker, = ax.plot([], [], 'go', markersize=10, label='起点')
            end_marker, = ax.plot([], [], 'ro', markersize=10, label='终点')

            # 设置图例
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

            ax.set_title(f"回合 {episode} 路径")
            ax.grid(True)
            plt.draw()
            plt.pause(pause_time)

        while not done:
            if render:
                # 动态渲染当前状态
                pass  # 已在下方的动态更新中处理

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_probs = policy_net(state_tensor)
            
            action = torch.argmax(action_probs, dim=1).item()
      
            next_state, reward, done, _ = env.step(action)
            path.append(env.current_pos)
            total_reward += reward
            state = next_state
            steps += 1

            if render:
                # 更新路径线
                path_x = [p[1] for p in path]
                path_y = [p[0] for p in path]
                path_line.set_data(path_x, path_y)

                # 更新起点和终点标记
                start_marker.set_data([path_x[0]], [path_y[0]])
                end_marker.set_data([path_x[-1]], [path_y[-1]])

                # 更新当前目标点标记
                # 清除之前的目标点标记（避免重复）
                for line in ax.get_lines():
                    if line.get_label() == '当前目标':
                        line.remove()

                for i, point in enumerate(env.waypoints):
                    if i == env.current_waypoint_idx:
                        ax.plot(point[1], point[0], 'r*', markersize=10, label='当前目标')
                    else:
                        if i == 0:
                            ax.plot(point[1], point[0], 'g*', markersize=10, label='目标点')
                        else:
                            ax.plot(point[1], point[0], 'g*', markersize=10)

                ax.set_title(f"回合 {episode} 路径")
                path_line.figure.canvas.draw()
                path_line.figure.canvas.flush_events()
                time.sleep(pause_time)

        total_steps += steps
    
        if env.current_waypoint_idx == len(env.waypoints) - 1 and env._is_at_target(env.waypoints[-1]):
            success_count += 1
        print(f"回合 {episode}: 总奖励 = {total_reward:.2f}, 步数 = {steps}, {'成功' if done and env.current_waypoint_idx == len(env.waypoints) - 1 else '失败'}")
        
        if render:
            plt.ioff()  # 关闭交互模式
            plt.show()

def main():
    # 设置全局种子
    SEED = 42
    set_seed(SEED)
    
    # 创建环境
    env = RLenv(map_file='map.csv', wall_distance=2, local_view=2, max_steps=1000)
 
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载策略网络
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = PolicyNetwork(state_dim, action_dim).to(device)
    
    # 加载训练好的模型
    policy_net.load_state_dict(torch.load("ppo_policy_net.pth", map_location=device))
    
    # 测试智能体
    test_agent(env, policy_net, device, num_episodes=3, render=True, seed=SEED, pause_time=0.1)  
    
    env.close()

if __name__ == "__main__":
    main()
