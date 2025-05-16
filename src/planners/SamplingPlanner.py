import numpy as np
import math
from spatialmath import SE3
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool


class Node:
    def __init__(self, theta, parent=None, cost=0.0):
        self.theta = theta
        self.parent = parent
        self.cost = cost


class SamplingPlanner:
    def __init__(self, obj, circle, arm_initial_pose):
        self.points = []
        self.dt = 0.05
        self.joint_angles = []
        self.dof = 3
        self.robot = obj
        self.stateFlag = False
        self.startCirclePoint = np.zeros(2)

        # 根据解析的circle来解出line
        self.init_pose = np.array(arm_initial_pose)
        self.radius = circle.radius
        self.center = np.array(circle.center)

        self.vecCenter2Init = self.center - self.init_pose
        self.distCenter2Init = np.linalg.norm(self.vecCenter2Init)
        self.vecInit2centerUnit = self.vecCenter2Init / self.distCenter2Init

        # 计算直线的一般式 Ax + By + C = 0
        self.lineA = self.init_pose[1] - self.center[1]  # y1 - y0
        self.lineB = self.center[0] - self.init_pose[0]  # x0 - x1
        self.lineC = self.init_pose[0] * self.center[1] - self.center[0] * self.init_pose[1]  # x1 y0 - x0 y1

        # 计算最近的候选点
        if self.distCenter2Init > 0:
            t1 = self.distCenter2Init - self.radius
            t2 = self.distCenter2Init + self.radius
            self.candidatePoint = self.init_pose + (t1 if abs(t1) < abs(t2) else t2) * self.vecInit2centerUnit
        else:
            self.candidatePoint = self.init_pose.copy()

    def calPoint2LineDist(self, point):
        xp = point[0]
        yp = point[1]
        return np.abs(self.lineA * xp + self.lineB * yp + self.lineC) / np.sqrt(self.lineA ** 2 + self.lineB ** 2)

    # 判断point是否在最短直线上
    def isOnLine(self, point, threshold=1e-2):
        # 计算点到直线的距离
        dist = self.calPoint2LineDist(point)
        if dist >= threshold:
            return False

        # 单位计算向量(point2init_pose)
        vecPoint2Init = point - self.init_pose

        # 计算投影
        # 若初始点在圆外 则不需要反向
        distThreshold = abs(np.linalg.norm(self.vecCenter2Init) - self.radius)
        if np.linalg.norm(self.vecCenter2Init) > self.radius:
            temp = np.dot(vecPoint2Init, self.vecInit2centerUnit)
        else:
            temp = np.dot(vecPoint2Init, -self.vecInit2centerUnit)

        return 0 <= temp <= distThreshold

    # 判断point是否在圆上
    def isOnCircle(self, point, threshold=1e-2):
        distance = np.linalg.norm(np.array(point) - self.center)
        return abs(distance - self.radius) < threshold

    # 关节角空间采样
    # def samplingTheta(self):
    #     return np.array([np.random.uniform(low, high) for low, high in zip(self.robot.qlim[0], self.robot.qlim[1])])
    def samplingTheta(self):
        try:
            if np.random.rand() < 0.5:
                q = self.robot.ikine_LM(
                    SE3(float(self.candidatePoint[0]), float(self.candidatePoint[1]), 0),
                    mask=np.array([1, 1, 1, 0, 0, 0]),
                    seed=88
                ).q
                if q is not None and len(q) == self.dof and not np.any(np.isnan(q)):
                    return q
        except:
            pass
        # fallback 到均匀采样
        low, high = self.robot.qlim
        return np.array([np.random.uniform(l, h) for l, h in zip(low, high)])

    # AI修改的
    # def samplingTheta(self):
    #     # 增加目标偏置采样比例
    #     if np.random.rand() < 0.5:  # 提高到50%概率采样目标方向
    #         if hasattr(self, 'candidatePoint'):
    #             # 在候选点周围生成高斯分布样本
    #             mean = self.robot.ikine_LM(SE3(*self.candidatePoint, 0), mask=[1, 1, 1, 0, 0, 0]).q
    #             return np.random.normal(mean, scale=0.2)  # 调整scale参数
    #     return np.array([np.random.uniform(low, high) for low, high in zip(self.robot.qlim[0], self.robot.qlim[1])])

    def fkinematics(self, theta):
        pose = self.robot.fkine(theta)
        return np.array([pose.t[0], pose.t[1]], dtype=np.float32)

    def calDist(self, theta1, theta2):
        point1 = self.fkinematics(theta1)
        point2 = self.fkinematics(theta2)
        return np.linalg.norm(point2 - point1)

    def near(self, nodes, sampleTheta, radius=0.5):
        return [node for node in nodes if self.calDist(node.theta, sampleTheta) < radius]

    def nearest(self, nodes, sampleTheta):
        dists = [self.calDist(node.theta, sampleTheta) for node in nodes]
        return nodes[np.argmin(dists)]

    def steer(self, fromTheta, toTheta, stepSize=0.05):
        fromPoint = self.fkinematics(fromTheta)
        toPoint = self.fkinematics(toTheta)
        vec = toPoint - fromPoint
        dist = np.linalg.norm(vec)
        unit = vec / dist
        if dist < stepSize:
            return toPoint
        else:
            return fromPoint + unit * stepSize

    # 代价函数是重要一环
    # 这里我们要考虑的是两个点
    # 一个是无障碍的时候 尽量在直线上
    # 另外一个是在圆上
    def costFunc(self, node):
        nodePoint = self.fkinematics(node.theta)
        distPoint2Line = self.calPoint2LineDist(nodePoint)
        distPoint2Circle = abs(np.linalg.norm(nodePoint - self.center) - self.radius)
        distPoint2Candidate = abs(np.linalg.norm(nodePoint - self.candidatePoint))

        print("The distPoint2Line : ", distPoint2Line)
        print("The distPoint2Circle : ", distPoint2Circle)
        print("The distPoint2Candidate : ", distPoint2Candidate)

        # 当前状态
        isOnCircle = self.isOnCircle(nodePoint)
        isOnLine = self.isOnLine(nodePoint)

        # 代价函数分为两阶段
        # 第一阶段为到达候选点
        print("State Flag : ", self.stateFlag)
        if isOnCircle:
            self.stateFlag = True

        # 第一阶段 走直线
        if not self.stateFlag:
            print("Current State: On Line")
            # 比例系数k
            k = 0.5
            # 放大系数k_
            k_ = 1
            costOnLine = k_ * (k * distPoint2Line + (1 - k) * distPoint2Candidate)
            print("The cost of the OnLine is: ", costOnLine)
            print("\n")
            return costOnLine
        # 第二阶段 走圆
        else:
            print("Current State: On Circle")
            if not hasattr(self, 'startCirclePoint'):
                self.startCirclePoint = nodePoint
                return distPoint2Circle - self.radius
            v1 = self.startCirclePoint - self.center
            v2 = nodePoint - self.center
            cosAngle = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)

            self.startCirclePoint = nodePoint
            costOnCircle = (distPoint2Circle - self.radius) + (- cosAngle) * self.radius
            print("The cost of the OnCircle is: ", costOnCircle)
            print("\n")
            return (distPoint2Circle - self.radius) + (- cosAngle) * self.radius

    def parallel_sample(self, _):
        """用于并行化的采样函数"""
        return self.samplingTheta()

    def edgeCost(self, node1, node2):
        # 有先后顺序 从node1到node2
        # 一般的RRT*的edgeCost都是两个点之间的距离
        # 但是我们这里不可行 需要重新考虑
        point1 = self.fkinematics(node1.theta)
        point2 = self.fkinematics(node2.theta)

        vec = point2 - point1
        dist = np.linalg.norm(vec)
        unit = vec / dist

        if self.isOnCircle(point1):
            self.stateFlag = True

        # 走圆阶段
        if self.stateFlag:
            # 计算叉乘
            # 作为沿圆的法线上运动的一个衡量
            # 越接近1越好
            vecPoint2Center = point1 - self.center
            distPoint2Center = np.linalg.norm(vecPoint2Center)
            unitPoint2Center = vecPoint2Center / distPoint2Center
            cross_product = np.cross(unitPoint2Center, unit)
            return cross_product
        # 走直线阶段
        else:
            # 计算点积
            # 作为和最短直线共线程度的一个衡量
            # 越接近1越好
            dot_product = np.dot(unit, self.vecInit2centerUnit)
            return dot_product

    def solution_without_collision(self):
        # 初始位置
        startNode = Node(self.robot.ikine_LM(
            SE3(float(self.init_pose[0]), float(self.init_pose[1]), 0),
            mask=np.array([1, 1, 1, 0, 0, 0]),
            seed=88
        ).q, cost=0.0)
        nodes = [startNode]
        goalCandidates = []
        maxIterations = 10
        with ThreadPool() as pool:
            for _ in range(maxIterations):
                print("The number of the iterations is ", _)
                samples = pool.map(self.parallel_sample, [None] * 5)
                for sample in samples:
                    # sample = self.samplingTheta()
                    nearestNode = self.nearest(nodes, sample)
                    newPoint = self.steer(nearestNode.theta, sample)

                    # 获得新节点
                    newNode = Node(self.robot.ikine_LM(
                        SE3(float(newPoint[0]), float(newPoint[1]), 0),
                        q0=nearestNode.theta,
                        mask=np.array([1, 1, 1, 0, 0, 0]),
                        seed=88,
                        ilimit=50, slimit=100, tol=1e-3
                    ).q)
                    newNode.parent = nearestNode
                    newNode.cost = nearestNode.cost + self.edgeCost(nearestNode, newNode)

                    neighbors = self.near(nodes, newNode.theta)

                    for neighbor in neighbors:
                        tempCost = neighbor.cost + self.edgeCost(neighbor, newNode)
                        if tempCost < newNode.cost:
                            newNode.parent = neighbor
                            newNode.cost = tempCost

                    for neighbor in neighbors:
                        tempCost = newNode.cost + self.edgeCost(newNode, neighbor)
                        if tempCost < neighbor.cost:
                            neighbor.parent = newNode
                            neighbor.cost = tempCost

                    nodes.append(newNode)

            if self.isOnCircle(self.fkinematics(newNode.theta)) or self.isOnLine(self.fkinematics(newNode.theta)):
                goalCandidates.append(newNode)
        goalPath = min(goalCandidates, key=lambda node: node.cost) if goalCandidates else None
        if goalPath:
            # 提取路径
            path = []
            node = goalPath
            while node:
                path.append(node.theta)
                node = node.parent
            return path[::-1]

    # def solution_without_collision(self):
    #     # 初始位置
    #     startNode = Node(self.robot.ikine_LM(
    #         SE3(float(self.init_pose[0]), float(self.init_pose[1]), 0),
    #         mask=np.array([1, 1, 1, 0, 0, 0]),
    #         seed=88
    #     ).q, cost=10)
    #
    #     nodes = [startNode]
    #
    #     maxIterations = 20
    #     best_node = None
    #     best_cost = -float('inf')
    #
    #     # 创建进程池
    #     with ThreadPool() as pool:
    #         for _ in range(maxIterations):
    #             print("The number of the iterations is ", _)
    #             samples = pool.map(self.parallel_sample, [None] * 5)
    #
    #             for sample in samples:
    #                 nearestNode = self.nearest(nodes, sample)
    #                 newPoint = self.steer(nearestNode.theta, sample)
    #
    #                 # 获得新节点
    #                 try:
    #                     new_theta = self.robot.ikine_LM(
    #                         SE3(float(newPoint[0]), float(newPoint[1]), 0),
    #                         q0=nearestNode.theta,
    #                         mask=np.array([1, 1, 1, 0, 0, 0]),
    #                         seed=88,
    #                         ilimit=50, slimit=100, tol=1e-3
    #                     ).q
    #                 except:
    #                     print("逆运动学解算失败")
    #                     continue  # 跳过逆运动学失败的情况
    #
    #                 newNode = Node(new_theta)
    #                 newNode.parent = nearestNode
    #                 newNode.cost = nearestNode.cost + self.costFunc(newNode)
    #
    #                 # 简化邻居处理
    #                 nodes.append(newNode)
    #
    #                 # if self.stateFlag:
    #                 #     best_cost = float("inf")
    #                 # 跟踪最佳节点
    #                 if newNode.cost < best_cost:
    #                     best_cost = newNode.cost
    #                     best_node = newNode
    #
    #     # 提取路径
    #     path = []
    #     node = best_node if best_node else (nodes[-1] if nodes else None)
    #     while node:
    #         path.append(node.theta)
    #         node = node.parent
    #     return path[::-1]  # 反转路径从起点到终点
