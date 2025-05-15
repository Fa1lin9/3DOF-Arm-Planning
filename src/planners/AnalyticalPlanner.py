import math
from spatialmath import SE3
import numpy as np


class AnalyticalPlanner:
    def __init__(self, obj, circle, arm_initial_pose, n):
        self.joint_angles = []
        self.path_points = None
        self.points = []
        self.robot = obj
        self.radius = circle.radius
        self.center = np.array(circle.center)

        self.init2center_unit = np.zeros(2)
        self.init_pose = np.array(arm_initial_pose)
        self.n = n

        self.area = self.radius * self.radius * math.pi
        self.circumference = 2 * self.radius * math.pi

    def generate_sample_points(self):
        # Step 1
        # generate the points of the circle
        start_angle = math.atan2(self.init2center_unit[1], self.init2center_unit[0])
        theta = np.linspace(start_angle, 2 * np.pi + start_angle, self.n, endpoint=False)
        circle_points = np.column_stack([
            self.radius * np.cos(theta) + self.center[0],
            self.radius * np.sin(theta) + self.center[1]
        ])
        circle_points = np.vstack((circle_points, circle_points[0]))
        # print("Circle points: ", circle_points)

        # Step 2
        # generate the points of the path from arm_initial_pose to the point closest to the circle
        dists = np.linalg.norm(circle_points - self.init_pose, axis=1)
        nearest_idx = np.argmin(dists)
        nearest_point = circle_points[nearest_idx]
        num_connect = 10
        connect_line = np.linspace(self.init_pose, nearest_point, num_connect)
        # print("Connect line: ", connect_line)

        self.path_points = np.vstack((connect_line, circle_points))
        # print("Path points: ", self.path_points)

        return self.path_points

    def solution_without_collision(self):
        # 计算从初始点到圆心的单位方向向量
        vec = np.array(self.init_pose) - np.array(self.center)
        norm = np.linalg.norm(vec)

        if norm == 0:
            self.init2center_unit = np.array([1.0, 0.0])
        else:
            self.init2center_unit = vec / norm

        # 生成轨迹 转成SE3的format 然后输出关节角空间的角度值
        trajectory = self.generate_sample_points()
        self.points = [SE3(x, y, 0) for x, y in trajectory]
        return self.get_joint_angle()

    def get_joint_angle(self):
        last_joint_angle = None
        for point in self.points:
            try:
                # 使用逆运动学求解器计算关节角度
                sol = self.robot.ikine_LM(
                    point,
                    q0=last_joint_angle,
                    mask=np.array([1, 1, 1, 0, 0, 0]),
                    seed=88
                    # joint_limits=True,
                )
                if sol.success:
                    self.joint_angles.append(sol.q)
                    last_joint_angle = sol.q
                else:
                    print(f"Failed to find a solution for point {point}")
            except Exception as e:
                print(f"Error during inverse kinematics for point {point}: {e}")
        return self.joint_angles

