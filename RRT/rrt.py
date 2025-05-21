import numpy as np
import random
import math

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRT:
    def __init__(self, env, step=1.0, max_iter=1000):
        self.env = env
        self.step = step  # 步长
        self.max_iter = max_iter  # 最大迭代次数
        self.nodes = []  # 树节点列表
        
    def plan(self, start, goal):
        # 坐标转换 (y,x) -> (x,y)
        start = (start[1], start[0])
        goal = (goal[1], goal[0])
        
        # 初始化树
        root = Node(start[0], start[1])
        self.nodes = [root]
        target = Node(goal[0], goal[1])
        
        # 主循环
        for _ in range(self.max_iter):
            # 采样新点
            if random.random() < 0.1:
                sample = (goal[0], goal[1])
            else:
                sample = self._sample()
            
            # 扩展树
            near = self._get_nearest(sample)
            new = self._extend_to(near, sample)
            if not new:
                continue
                
            self.nodes.append(new)
            
            # 尝试连接目标
            if self._dist(new, target) < self.step:
                if self._is_valid_edge(new, target):
                    target.parent = new
                    path = self._get_path(target)
                    return self._smooth_path(path)
        
        print("No path found")
        return []
    
    def _sample(self):
        # 在地图内采样,带边界保护
        m = 1  # 边界margin
        x = random.randint(m, self.env.width-1-m)
        y = random.randint(m, self.env.height-1-m)
        
        # 偏向目标点采样
        if random.random() < 0.2:
            goal = self.env.waypoints[self.env.current_waypoint_idx]
            r = random.randint(1, 3)
            a = random.uniform(0, 2 * math.pi)
            x = int(goal[1] + r * math.cos(a))
            y = int(goal[0] + r * math.sin(a))
            x = min(max(m, x), self.env.width-1-m)
            y = min(max(m, y), self.env.height-1-m)
        
        return (x, y)
    
    def _get_nearest(self, point):
        # 返回最近节点
        dists = [self._point_dist(n, point) for n in self.nodes]
        return self.nodes[np.argmin(dists)]
    
    def _dist(self, n1, n2):
        # 节点间欧氏距离
        return math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
    
    def _point_dist(self, node, point):
        # 节点到点的距离
        return math.sqrt((node.x - point[0])**2 + (node.y - point[1])**2)
    
    def _extend_to(self, src, dst):
        # 从src向dst扩展
        dx = dst[0] - src.x
        dy = dst[1] - src.y
        d = math.sqrt(dx*dx + dy*dy)
        
        if d == 0:
            return None
            
        # 自适应步长
        step = min(self.step, d)
        
        # 计算新节点位置
        dx = dx/d * step
        dy = dy/d * step
        x = int(round(src.x + dx))
        y = int(round(src.y + dy))
        
        # 边界检查
        x = min(max(1, x), self.env.width-2)
        y = min(max(1, y), self.env.height-2)
        
        # 碰撞检查
        new = Node(x, y)
        if not self._is_valid_edge(src, new):
            return None
            
        new.parent = src
        return new
    
    def _is_valid_edge(self, n1, n2):
        # 检查路径有效性
        points = max(int(self._dist(n1, n2) * 3), 3)
        
        for i in range(points + 1):
            t = i / points
            x = int(round(n1.x + t*(n2.x - n1.x)))
            y = int(round(n1.y + t*(n2.y - n1.y)))
            
            # 检查周围8邻域
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if not self.env._is_valid_position((y + dy, x + dx)):
                        return False
        return True
    
    def _get_path(self, target):
        # 提取路径
        path = []
        curr = target
        while curr:
            path.append((int(curr.y), int(curr.x)))
            curr = curr.parent
        return list(reversed(path))
    
    def _smooth_path(self, path):
        # 路径平滑
        if len(path) <= 2:
            return path
        
        smooth = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # 寻找可直连的最远点
            for j in range(len(path)-1, i, -1):
                if self._is_valid_edge(Node(path[i][1], path[i][0]), 
                                     Node(path[j][1], path[j][0])):
                    smooth.append(path[j])
                    i = j
                    break
            else:
                i += 1
                if i < len(path):
                    smooth.append(path[i])
        
        return smooth 