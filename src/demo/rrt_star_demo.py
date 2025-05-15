import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 机械臂参数
LINK_LENGTHS = [1.0, 1.0, 1.0]
JOINT_LIMITS = [(-np.pi, np.pi)] * 3

# 圆轨迹目标
CIRCLE_CENTER = np.array([1.5, 0.0])
CIRCLE_RADIUS = 0.5

# RRT* 参数
STEP_SIZE = 0.1
MAX_ITER = 1000
NEIGHBOR_RADIUS = 0.4

class Node:
    def __init__(self, theta, parent=None, cost=0.0):
        self.theta = theta
        self.parent = parent
        self.cost = cost

def forward_kinematics(theta):
    x = y = 0
    total_theta = 0
    for i in range(3):
        total_theta += theta[i]
        x += LINK_LENGTHS[i] * np.cos(total_theta)
        y += LINK_LENGTHS[i] * np.sin(total_theta)
    return np.array([x, y])

def sample_theta():
    return np.array([np.random.uniform(low, high) for low, high in JOINT_LIMITS])

def end_effector_distance(theta1, theta2):
    p1 = forward_kinematics(theta1)
    p2 = forward_kinematics(theta2)
    return np.linalg.norm(p1 - p2)

def nearest(tree, sample):
    dists = [np.linalg.norm(node.theta - sample) for node in tree]
    return tree[np.argmin(dists)]

def near(tree, sample, radius):
    return [node for node in tree if np.linalg.norm(node.theta - sample) < radius]

def steer(from_theta, to_theta, step_size):
    direction = to_theta - from_theta
    dist = np.linalg.norm(direction)
    if dist < step_size:
        return to_theta
    return from_theta + direction / dist * step_size

def cost_to_goal(node):
    end_pos = forward_kinematics(node.theta)
    return abs(np.linalg.norm(end_pos - CIRCLE_CENTER) - CIRCLE_RADIUS)

def plot_arm(theta, color='b'):
    x, y = [0], [0]
    total_theta = 0
    for i in range(3):
        total_theta += theta[i]
        x.append(x[-1] + LINK_LENGTHS[i] * np.cos(total_theta))
        y.append(y[-1] + LINK_LENGTHS[i] * np.sin(total_theta))
    plt.plot(x, y, f'{color}-o', linewidth=2)

# 初始化树
start_theta = np.array([0.0, 0.0, 0.0])
start_node = Node(start_theta)
tree = [start_node]
goal_candidates = []

# RRT* 主循环
for _ in range(MAX_ITER):
    # 获得随机采样点
    sample = sample_theta()
    # 在已有的节点数中寻找距离采样点最近的节点
    nearest_node = nearest(tree, sample)
    # 拓展新节点
    new_theta = steer(nearest_node.theta, sample, STEP_SIZE)
    # 赋值：角度 父节点 代价
    new_node = Node(new_theta)
    new_node.parent = nearest_node
    new_node.cost = nearest_node.cost + end_effector_distance(nearest_node.theta, new_theta)

    # 根据新节点的角度在附近寻找neighbor
    neighbors = near(tree, new_theta, NEIGHBOR_RADIUS)

    # 最优父节点选择
    # Note : 节点的选择与代价强相关 代价的构造是重要一环
    for neighbor in neighbors:
        # 计算neighbor到新节点的代价
        tentative_cost = neighbor.cost + end_effector_distance(neighbor.theta, new_theta)
        # 寻找代价最小的父节点
        if tentative_cost < new_node.cost:
            new_node.parent = neighbor
            new_node.cost = tentative_cost

    # Rewire 操作
    for neighbor in neighbors:
        if new_node.cost + end_effector_distance(new_node.theta, neighbor.theta) < neighbor.cost:
            neighbor.parent = new_node
            neighbor.cost = new_node.cost + end_effector_distance(new_node.theta, neighbor.theta)

    tree.append(new_node)

    # 如果靠近圆周，加入候选
    if cost_to_goal(new_node) < 0.05:
        goal_candidates.append(new_node)

# 从候选路径中找代价最小的
goal_path = min(goal_candidates, key=lambda node: node.cost) if goal_candidates else None

# 可视化 + 动画
theta_circle = np.linspace(0, 2*np.pi, 100)
circle_x = CIRCLE_CENTER[0] + CIRCLE_RADIUS * np.cos(theta_circle)
circle_y = CIRCLE_CENTER[1] + CIRCLE_RADIUS * np.sin(theta_circle)

if goal_path:
    print(f"找到路径，总节点数：{len(tree)}，末端误差：{cost_to_goal(goal_path):.4f}")

    # 提取路径
    path = []
    node = goal_path
    while node:
        path.append(node.theta)
        node = node.parent
    path.reverse()

    # 动画绘制
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title("RRT*")

    ax.plot(circle_x, circle_y, 'g--', label="Target Circle")
    arm_line, = ax.plot([], [], 'r-o', linewidth=2)
    trace_line, = ax.plot([], [], 'b-', linewidth=1, alpha=0.5)
    trace_points = [[], []]

    def init():
        arm_line.set_data([], [])
        trace_line.set_data([], [])
        return arm_line, trace_line

    def animate(i):
        theta = path[i]
        x, y = [0], [0]
        total_theta = 0
        for j in range(3):
            total_theta += theta[j]
            x.append(x[-1] + LINK_LENGTHS[j] * np.cos(total_theta))
            y.append(y[-1] + LINK_LENGTHS[j] * np.sin(total_theta))
        arm_line.set_data(x, y)
        trace_points[0].append(x[-1])
        trace_points[1].append(y[-1])
        trace_line.set_data(trace_points[0], trace_points[1])
        return arm_line, trace_line

    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(path), interval=200, blit=True, repeat=False)

    plt.legend()
    plt.show()
else:
    print("未能找到画圆轨迹的路径。")
