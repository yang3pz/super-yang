import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym import spaces
import random
import math
import os
from matplotlib import colors  
from rrt import RRT  

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 

class RLenv(gym.Env):
    def __init__(self, map_file='map.csv', wall_distance=2, local_view=2, max_steps=1000, alpha=1.0):
        super(RLenv, self).__init__()
        
        self.map = self.load_map(map_file)
        self.height, self.width = self.map.shape

    
        self.wall_distance = wall_distance
        self.start_point = (2, 2)  # 起点坐标 (y, x)，修改了这个参数，下面的waypoints也要修改一样的 保证起点是一个就行
        
        # 设置waypoints为起点和终点
        self.waypoints = [
            (2, 2),    # 起点
            (17, 17),  # 终点
        ]

        # 确保起点和目标点是可行的位置
        for point in self.waypoints:
            if not self._is_valid_position(point):
                raise ValueError(f"目标点{point}是无效的位置，无法通过。请检查地图和wall_distance设置。")

        # 当前目标点索引
        self.current_waypoint_idx = 1  # 从第一个目标点开始

        # 设置动作空间（8个方向）
        self.action_space = spaces.Discrete(8)

        # 定义8个方向的移动向量 (y, x)
        self.actions = [
            (-1, 0),   # 上
            (-1, 1),   # 右上
            (0, 1),    # 右
            (1, 1),    # 右下
            (1, 0),    # 下
            (1, -1),   # 左下
            (0, -1),   # 左
            (-1, -1)   # 左上
        ]

        # 设置观察空间
        # 状态空间：当前位置(y, x), 当前目标点(y, x), 下一个全局路径点(y, x), local_view * local_view grid
        self.local_view = local_view
        self.observation_space = spaces.Box(
            low=0,
            high=3,  # 围墙=1, 凸起=2, 障碍物=3
            shape=(6 + (2 * local_view + 1) ** 2,),
            dtype=np.float32
        )

        self.max_steps = max_steps
        
        # 定义全局路径规划器，传递alpha参数
        self.planner = RRT(self, step=1.0, max_iter=1000)  # RRT路径规划器
        
        # 初始化随机数生成器
        self.seed(seed=None)

    def seed(self, seed=None):
        """
        设置环境的随机种子。

        参数:
        - seed (int): 要设置的种子值。

        返回:
        - list: 设置的种子值。
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def load_map(self, filename):
        if not os.path.exists(filename):
            # 如果map.csv不存在，创建一个默认地图
            print(f"地图文件 '{filename}' 未找到，创建默认地图。")
            map_matrix = np.zeros((20, 20))
            # 添加围墙
            map_matrix[0, :] = 1
            map_matrix[-1, :] = 1
            map_matrix[:, 0] = 1
            map_matrix[:, -1] = 1
            # 添加凸起
            map_matrix[0:4, 9:12] = 2  # 凸起_上
            map_matrix[18:20, 9:12] = 2  # 凸起_下
            map_matrix[9:12, 0:5] = 2  # 凸起_左
            map_matrix[9:15, 16:20] = 2  # 凸起_右
            # 添加障碍物
            map_matrix[5:7, 5:7] = 3
            map_matrix[15:17, 15:17] = 3
            map_matrix[10:12, 10:12] = 3
            map_matrix[10:12, 14:16] = 3
            map_matrix[14:16, 10:12] = 3
            return map_matrix

        # 读取CSV文件
        df = pd.read_csv(filename, comment='#')

        # 创建20x20的空地图
        map_matrix = np.zeros((20, 20))

        # 元素类型到数值的映射
        element_values = {
            '围墙': 1,
            '凸起': 2,
            '障碍物': 3
        }

        # 填充地图
        for _, row in df.iterrows():
            # 获取元素类型（取第一个下划线前的部分）
            element_type = row['元素名称'].split('_')[0]
            value = element_values.get(element_type, 0)

            # 填充对应区域
            x_start = int(row['起始X'])
            y_start = int(row['起始Y'])
            width = math.ceil(float(row['宽度']))  # 使用 math.ceil 向上取整
            height = math.ceil(float(row['高度']))  # 确保高度为整数

            print(f"加载元素: {row['元素名称']}, 起始X: {x_start}, 起始Y: {y_start}, 宽度: {width}, 高度: {height}, 值: {value}")

            # 确保不超出地图范围
            x_end = min(x_start + width, map_matrix.shape[1])
            y_end = min(y_start + height, map_matrix.shape[0])

            map_matrix[y_start:y_end, x_start:x_end] = value

        # 打印地图中各元素的数量
        unique, counts = np.unique(map_matrix, return_counts=True)
        print("地图元素分布:", dict(zip(unique, counts)))

        return map_matrix

    def reset(self):
        """重置环境"""
        self.current_pos = self.start_point
        self.current_waypoint_idx = 1  # 从第一个目标点开始
        self.steps = 0
        self.global_path = self.planner.plan(self.current_pos, self.waypoints[self.current_waypoint_idx])

        # 验证凸起区域
        print("凸起_上区域值:")
        print(self.map[0:4, 9:12])

        print("凸起_下区域值:")
        print(self.map[18:20, 9:12])

        print("凸起_左区域值:")
        print(self.map[9:12, 0:5])

        print("凸起_右区域值:")
        print(self.map[9:12, 18:20])

        return self._get_obs()

    def _get_local_view(self):
        """获取智能体周围的局部视野"""
        y, x = self.current_pos
        lv = self.local_view
        # 定义视野范围
        y_min = max(y - lv, 0)
        y_max = min(y + lv + 1, self.height)
        x_min = max(x - lv, 0)
        x_max = min(x + lv + 1, self.width)

        # 提取局部地图
        local_map = self.map[y_min:y_max, x_min:x_max]

        # 如果视野不足，填充0
        pad_y_before = max(lv - y, 0)
        pad_y_after = max((y + lv + 1) - self.height, 0)
        pad_x_before = max(lv - x, 0)
        pad_x_after = max((x + lv + 1) - self.width, 0)

        if pad_y_before > 0 or pad_y_after > 0 or pad_x_before > 0 or pad_x_after > 0:
            local_map = np.pad(
                local_map,
                ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after)),
                'constant',
                constant_values=0
            )

        return local_map.flatten()

    def _get_obs(self):
        """获取当前观察，包括当前位置、当前目标点、下一个全局路径点和局部视野"""
        # 获取全局路径的下一个点
        if len(self.global_path) > 1:
            next_point = self.global_path[1]
        else:
            next_point = self.waypoints[self.current_waypoint_idx]

        # 获取局部视野
        local_view = self._get_local_view()

        return np.concatenate((
            np.array([
                self.current_pos[0],
                self.current_pos[1],
                self.waypoints[self.current_waypoint_idx][0],
                self.waypoints[self.current_waypoint_idx][1],
                next_point[0],
                next_point[1]
            ], dtype=np.float32),
            local_view
        ))

    def check_path_blocked(self, start, end):
        """检查两点之间是否有障碍物"""
        y1, x1 = start
        y2, x2 = end
        dx = x2 - x1
        dy = y2 - y1
        distance = max(abs(dx), abs(dy))  # 使用最大步长来计算距离

        if distance == 0:  # 如果起点和终点是同一个点，返回 False
            return False

        for i in range(1, int(distance)):  # 遍历路径上的每个点，避免起点
            x = int(x1 + dx * i / distance)
            y = int(y1 + dy * i / distance)

            # 如果路径上的某个点是障碍物，则返回 True
            if self._is_collision((y, x)):  # 使用 (y, x)
                print(f"路径检查: ({y}, {x}) 是障碍物")
                return True

        return False

    def step(self, action):
        """执行一步动作"""
        self.steps += 1

        # 获取动作对应的方向
        direction = self.actions[action]

        # 计算新位置
        new_pos = (
            self.current_pos[0] + direction[0],
            self.current_pos[1] + direction[1]
        )

        # 检查是否有效移动
        if self._is_valid_position(new_pos):
            self.current_pos = new_pos
        else:
            # 无效移动（如碰到墙或障碍物），保持原地
            new_pos = self.current_pos

        # 计算与当前目标点的距离
        current_distance = self._distance(self.current_pos, self.waypoints[self.current_waypoint_idx])

        # 初始化回报和完成标志
        done = False
        reward = -0.1  # 基础移动成本

        # 如果到达当前目标点
        if self._is_at_target(self.waypoints[self.current_waypoint_idx]):
            reward += 100.0  # 到达目标点奖励
            if self.current_waypoint_idx < len(self.waypoints) - 1:
                self.current_waypoint_idx += 1
                # 重新规划到下一个目标点的路径
                self.global_path = self.planner.plan(self.current_pos, self.waypoints[self.current_waypoint_idx])
            else:
                done = True  # 已回到起点

        # 计算下一步与当前目标点的距离
        next_distance = self._distance(self.current_pos, self.waypoints[self.current_waypoint_idx])

        # 奖励函数：奖励减少距离，惩罚增加距离
        reward += (current_distance - next_distance) * 1.0

        # 如果靠近目标点，给予额外奖励
        reward += -0.01 * next_distance  # 奖励接近目标点

        # 如果碰到障碍物（当前移动位置不是原地）
        if new_pos != self.current_pos and self._is_collision(new_pos):
            done = True
            reward -= 100.0  # 碰撞惩罚

        # 超出步数限制
        if self.steps >= self.max_steps:
            done = True
            reward -= 100.0  # 步数限制惩罚

        # 获取下一状态
        next_state = self._get_obs()

        return next_state, reward, done, {}

    def _is_valid_position(self, pos):
        """检查位置是否有效，并为不同类型的障碍物分配不同的成本"""
        y, x = pos
        if 0 <= x < self.width and 0 <= y < self.height:
            cell = self.map[y, x]
            if cell == 0:
                return True
            elif cell == 1:
                print(f"位置 {pos} 是围墙，不可通过。")
                return False  # 围墙，不可通过
            elif cell == 2:
                print(f"位置 {pos} 是凸起，不可通过。")
                return False  # 凸起，也视为障碍物
            elif cell == 3:
                print(f"位置 {pos} 是障碍物，不可通过。")
                return False  # 其他障碍物
        print(f"位置 {pos} 超出地图范围或无效。")
        return False

    def _is_collision(self, pos):
        """检查是否碰撞"""
        y, x = pos
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.map[y, x] != 0
        return True

    def _is_at_target(self, target, threshold=0.5):
        """检查是否到达目标点"""
        return (abs(self.current_pos[0] - target[0]) < threshold and
                abs(self.current_pos[1] - target[1]) < threshold)

    def _distance(self, pos1, pos2):
        """计算两个位置之间的切比雪夫距离"""
        y1, x1 = pos1
        y2, x2 = pos2
        return max(abs(y1 - y2), abs(x1 - x2))

    def verify_path(self, path):
        """验证路径中是否包含任何无效的位置"""
        for pos in path:
            if not self._is_valid_position(pos):
                print(f"路径中包含无效位置: {pos}")
                return False
        print("路径验证通过，所有位置均为有效通行点。")
        return True

    def render(self, mode='human'):
        """渲染环境"""
        plt.clf()
        # 定义颜色映射：0-空地, 1-围墙, 2-凸起, 3-障碍物
        cmap = colors.ListedColormap(['white', 'black', 'red', 'blue'])
        bounds = [0,1,2,3,4]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        plt.imshow(self.map, cmap=cmap, norm=norm, origin='upper')

        for i, point in enumerate(self.waypoints):
            if i == self.current_waypoint_idx:
                plt.plot(point[1], point[0], 'r*', markersize=10, label='当前目标')
            else:
                if i == 0:
                    plt.plot(point[1], point[0], 'g*', markersize=10, label='目标点')
                else:
                    plt.plot(point[1], point[0], 'g*', markersize=10)

        if self.global_path and len(self.global_path) > 0:
            path_x = [p[1] for p in self.global_path]
            path_y = [p[0] for p in self.global_path]
            plt.plot(path_x, path_y, 'y--', linewidth=2, label='全局路径')

        plt.plot(self.current_pos[1], self.current_pos[0], 'bo', markersize=10, label='智能体')

        plt.grid(True)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.pause(0.001)
        plt.draw()


def test_rrt_to_goal():  # 替换原来的 test_astar_to_goal
    env = RLenv(map_file='map.csv')
    start = (2, 2)  # (y, x)
    goal = (2, 17)  # (y, x)
    path = env.planner.plan(start, goal)
   
    if path:
        env.verify_path(path)
    
    plt.figure(figsize=(6,6))
    cmap = colors.ListedColormap(['white', 'black', 'red', 'blue'])
    bounds = [0,1,2,3,4]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(env.map, cmap=cmap, norm=norm, origin='upper')
    
    if path:
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        plt.plot(path_x, path_y, 'y--', linewidth=2, label='RRT路径')
    plt.plot(start[1], start[0], 'bo', markersize=10, label='起点')
    plt.plot(goal[1], goal[0], 'ro', markersize=10, label='目标点')
    plt.legend()
    plt.title("RRT路径规划测试")
    plt.show()


if __name__ == "__main__":
    test_rrt_to_goal()  # 替换原来的 test_astar_to_goal()
