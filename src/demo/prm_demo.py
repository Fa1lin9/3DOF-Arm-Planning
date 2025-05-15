import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from matplotlib.patches import Rectangle


def is_point_in_obstacle(point, box_list):
    """检测点是否在障碍物矩形内 (轴对齐矩形检测)"""
    x, y = point
    for (x1, y1, x2, y2) in box_list:
        if (min(x1, x2) <= x <= max(x1, x2)) and (min(y1, y2) <= y <= max(y1, y2)):
            return True
    return False


def is_collision_line(p1, p2, box_list):
    """改进碰撞检测：分10段采样检测线段与障碍物相交"""
    # 判断路径是否与障碍物相交
    for t in np.linspace(0, 1, 10):
        # 这里是在采样路径上的点 看这些点是否在障碍物里
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        if is_point_in_obstacle((x, y), box_list):
            return True
    return False


def prm_algorithm(height, width, box_list, start, end, n_samples=200, k_neighbors=8, radius=50):
    """支持起终点的PRM主函数"""
    # 验证起终点合法性
    if is_point_in_obstacle(start, box_list) or is_point_in_obstacle(end, box_list):
        raise ValueError("Start/End points are in obstacle!")

    # 1. 包含起终点的采样点集合
    samples = [start, end]  # 保证起终点存在
    while len(samples) < n_samples + 2:
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        if not is_point_in_obstacle((x, y), box_list):
            samples.append((x, y))

    # 2. 构建KDTree加速邻域搜索
    kd_tree = KDTree(samples)

    # 3. 创建网络图
    # 在网络图中可视化samples的点集
    G = nx.Graph()
    for i, pos in enumerate(samples):
        G.add_node(i, pos=pos)

    # 4. 连接所有节点（含起终点）
    for i, point in enumerate(samples):
        # 使用KDTree寻找目标点附近neighbor 即距离point最近的点集
        # 其中需要+1是因为排除自身
        distances, indices = kd_tree.query(point, k=k_neighbors + 1)
        for idx in indices[1:]:  # 排除自身
            neighbor = samples[idx]
            # 检验neighbor是否在障碍物内
            # 然后检查是否在预设半径内 即让neighbor距离point的距离都在radius内
            if not is_collision_line(point, neighbor, box_list):
                dist = np.linalg.norm(np.array(point) - np.array(neighbor))
                if dist < radius:
                    G.add_edge(i, idx, weight=dist)
    return G


def query_path(G, start, end, box_list):
    """在PRM图中查询起终点路径"""
    # 找到起终点对应的节点ID
    start_id = None
    end_id = None
    for node in G.nodes(data='pos'):
        if np.allclose(node[1], start):
            start_id = node[0]
        if np.allclose(node[1], end):
            end_id = node[0]

    # 使用Dijkstra算法寻路
    try:
        path_nodes = nx.shortest_path(G, start_id, end_id, weight='weight')
        path = [G.nodes[n]['pos'] for n in path_nodes]
        return path
    except nx.NetworkXNoPath:
        return None


def visualize_prm(G, box_list, start, end, path=None):
    """增强可视化：显示起终点与规划路径"""
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # 绘制障碍物
    for (x1, y1, x2, y2) in box_list:
        ax.add_patch(Rectangle(
            (min(x1, x2), min(y1, y2)),
            abs(x2 - x1), abs(y2 - y1),
            edgecolor='black', facecolor='gray', alpha=0.7
        ))

    # 绘制所有节点与边
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color='blue', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=0.5, ax=ax)

    # 高亮起终点
    nx.draw_networkx_nodes(G, pos, nodelist=[0], node_color='green', node_size=100, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[1], node_color='red', node_size=100, ax=ax)

    # 绘制规划路径
    if path:
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)

    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.grid(True)
    plt.title("PRM Path Planning")
    plt.show()


if __name__ == "__main__":
    # 定义地图与障碍物
    height, width = 100, 100
    box_list = [
        (20, 20, 40, 40),  # 障碍物1,对角点坐标
        (60, 10, 80, 90),  # 障碍物2
        (10, 70, 30, 90)  # 障碍物3
    ]
    start = (5, 5)  # 起点坐标
    end = (95, 80)  # 终点坐标

    # 生成PRM图
    prm_graph = prm_algorithm(height, width, box_list, start, end, n_samples=150)

    # 查询路径
    path = query_path(prm_graph, start, end, box_list)

    # 可视化结果
    visualize_prm(prm_graph, box_list, start, end, path)
