import numpy as np
import math
from spatialmath import SE3
import qpsolvers as qp


class OptimPlanner:
    def __init__(self, obj, circle, arm_initial_pose):
        self.points = []
        self.dt = 0.05
        self.joint_angles = []
        self.dof = 3
        self.robot = obj
        self.radius = circle.radius
        self.center = np.array(circle.center)
        # print("center : ", self.center)
        self.init2center_unit = np.zeros(2)

        self.init_pose = np.array(arm_initial_pose)
        # print(self.init_pose)
        try:
            sol = self.robot.ikine_LM(
                SE3(float(self.init_pose[0]), float(self.init_pose[1]), 0),
                mask=np.array([1, 1, 1, 0, 0, 0]),
                seed=88
                # joint_limits=True,
            )
            if sol.success:
                self.init_angle = sol.q
            else:
                print(f"Failed to find a solution for point {self.init_pose}")
        except Exception as e:
            print(f"Error during inverse kinematics for point {self.init_pose}: {e}")

        self.current_pose = np.array(self.init_pose, dtype=np.float64)
        self.robot.q = self.init_angle

        self.dist_init2center = np.linalg.norm(self.init_pose - self.center)

        self.area = self.radius * self.radius * math.pi
        self.circumference = 2 * self.radius * math.pi

    # 速度函数 其中dist和threshold均大于0
    # 当dist > threshold时 速度随dist - threshold的减小而减小
    # 当dist < threshold时 速度为定值
    # 分界点平滑过渡
    def velocity_func(self, dist, threshold, v_min=0.4, v_max=2.0, k=1.0):
        if dist <= threshold:
            return v_min
        else:
            # 使用平滑sigmoid型函数进行过渡
            delta = dist - threshold
            return v_max - (v_max - v_min) * math.exp(-k * delta)

    def get_expected_velocity(self):
        # 主要思路是这样的
        # 根据目前末端位置与圆心的距离
        # 在机械臂末端以圆的法向和径向建系 那么末端速度就是两者的分量
        # 当末端在圆上时 只给径向速度 且距离圆越近速度越小
        # 当末端不在圆上时 只给法向速度

        direction_vector = self.center - self.current_pose
        dist_current2center = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)

        # 末端在圆心上时 圆上点距离原点距离都为self.radius 这时随便找个目标点
        if dist_current2center < 1e-4:
            direction_unit = np.array([1, 0])
        else:
            direction_unit = direction_vector / dist_current2center

        vx, vy, w = 0.0, 0.0, 0.0
        # 若与圆上的最近点的距离大于一定值 则需要跟踪到圆上 只提供径向上的速度
        # 反之 则说明在圆上 则提供法向上的速度
        threshold = 1e-2
        velocity = self.velocity_func(abs(dist_current2center - self.radius), threshold)
        if abs(dist_current2center - self.radius) > threshold:
            # print("径向上速度")
            v_r = velocity * (1 if (dist_current2center - self.radius) > 0 else -1)
            velocity_vector = v_r * direction_unit
            vx, vy = velocity_vector[0], velocity_vector[1]
        else:
            # print("法向上速度")
            # 顺时针绕圈
            # v_n = velocity * np.array([-direction_unit[1], direction_unit[0]])
            # 逆时针绕圈
            v_n = velocity * np.array([direction_unit[1], -direction_unit[0]])
            velocity_vector = v_n
            vx, vy = velocity_vector[0], velocity_vector[1]
            # w = np.linalg.norm(v_n) / self.radius
        return np.array([vx, vy, w])
        # return np.array([0.5, 0, 0])

    def using_jacobian(self):
        jacobe = self.robot.jacob0(self.robot.q)[:2, :self.dof]
        dq = np.linalg.pinv(jacobe[:2, :]) @ self.get_expected_velocity()[:2]
        return dq

    def qp_solve_equity_constraint(self):
        slack_num = 2
        jacobe = self.robot.jacob0(self.robot.q)
        expected_v = self.get_expected_velocity()
        jacobe_ = np.vstack((jacobe[:2, :self.dof], jacobe[-1, :self.dof]))

        # Define the Q
        Q = np.eye(self.dof + slack_num)
        Q[0, 0] *= 1
        Q[1, 1] *= 10
        Q[2, 2] *= 100
        # 惩罚给小了会不按照期望的速度走
        Q[self.dof:, self.dof:] *= 1000

        # Define the f
        c = np.zeros(self.dof + slack_num)
        # penalty = np.exp(- (self.robot.q[2] / 0.1) ** 2)  # 越接近0越大
        # c[2] = -0.5 * penalty

        # 等式约束
        Aeq = np.c_[jacobe[:2, :self.dof], np.eye(slack_num)]
        beq = expected_v[:2]

        # 不等式约束
        Ain = np.zeros((self.dof + slack_num, self.dof + slack_num))
        bin = np.zeros(self.dof + slack_num)
        use_damper = False
        if use_damper:
            ps = 0.1
            pi = 0.9
            Ain[: self.dof, : self.dof], bin[: self.dof] = self.robot.joint_velocity_damper(ps, pi, self.dof)

        # 上下限
        qdlim = 2.0
        slack_limit = 10
        lb = -np.r_[qdlim * np.ones(self.dof), slack_limit * np.ones(slack_num)]
        ub = np.r_[qdlim * np.ones(self.dof), slack_limit * np.ones(slack_num)]

        # solve QP
        try:
            result = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='quadprog')
            qd = result[:self.dof]  # 提取关节速度部分
        except Exception as e:
            print("QP求解失败:", e)
            qd = np.zeros(self.dof)
        qd = qd[: self.dof]
        return qd

    def qp_solve_target_func(self):
        expected_v = self.get_expected_velocity()[:2]
        jacobe = self.robot.jacob0(self.robot.q)[:2, :self.dof]
        # jacobe = np.vstack((jacobe[:2, :self.dof], jacobe[-1, :self.dof]))

        # Define the H
        H = jacobe.T @ jacobe
        epsilon = 1e-4
        H += epsilon * np.eye(H.shape[0])

        # Define the f
        f = - jacobe.T @ expected_v

        # 不等式约束 速度阻尼器
        use_damping = False
        Ain = np.zeros((self.dof, self.dof))
        bin = np.zeros(self.dof)
        if use_damping:
            # ps
            # 定义是 关节允许接近极限的最小距离，低于此距离时速度必须降为0。
            ps = 0.01
            # pi是关键值
            # 定义是 速度阻尼开始生效的阈值，关节进入此范围后速度按比例衰减。
            pi = 0.2
            Ain[: self.dof, : self.dof], bin[: self.dof] = (
                self.robot.joint_velocity_damper(ps, pi, self.dof))

        # 上下界
        qdlim = 4.0
        lb = -np.r_[qdlim * np.ones(3)]
        ub = np.r_[qdlim * np.ones(3)]

        # solve QP
        qd = qp.solve_qp(H, f, Ain, bin, lb=lb, ub=ub, solver='scs')
        qd = qd[: self.dof]
        return qd

    def qp_solve_target_func_use_slack(self):
        slack_num = 2
        lambda_slack = 10  # 松弛变量惩罚系数

        expected_v = self.get_expected_velocity()[:2]
        jacobe = self.robot.jacob0(self.robot.q)[:2, :self.dof]

        # Define the H
        H = np.zeros((self.dof + slack_num, self.dof + slack_num))
        H[:self.dof, :self.dof] = jacobe.T @ jacobe
        # 确保H正定
        epsilon = 1e-3
        H[:self.dof, :self.dof] += epsilon * np.eye(self.dof)
        H[self.dof:, self.dof:] = lambda_slack * np.eye(slack_num)

        # Define the f
        f = np.zeros(self.dof + slack_num)
        f[:self.dof] = - jacobe.T @ expected_v
        f[self.dof:] = -expected_v

        Ain = np.zeros((self.dof + slack_num, self.dof + slack_num))
        bin = np.zeros(self.dof + slack_num)

        ps = 0.05
        pi = 0.1
        Ain[: self.dof, : self.dof], bin[: self.dof] = (
            self.robot.joint_velocity_damper(ps, pi, self.dof))

        qdlim = 4.0
        lb = -np.r_[qdlim * np.ones(3), 1 * np.ones(slack_num)]
        ub = np.r_[qdlim * np.ones(3), 1 * np.ones(slack_num)]

        # solve QP
        qd = qp.solve_qp(H, f, Ain, bin, lb=lb, ub=ub, solver='scs')
        qd = qd[: self.dof]
        return qd

    def null_space(self):
        expected_v = self.get_expected_velocity()[:2]
        jacobe = self.robot.jacob0(self.robot.q)[:2, :self.dof]
        J_pinv = np.linalg.pinv(jacobe[:2, :])
        print(f"Shape of the J_pinv", J_pinv.shape)
        print(f"Shape of the J", jacobe.shape)
        # 主任务
        qd_primary = J_pinv @ expected_v
        # 副任务
        qd_null = -self.robot.q * 0.1
        null_proj = np.eye(self.dof) - J_pinv @ jacobe[:2, :]
        qd = qd_primary + null_proj @ qd_null
        return qd

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

    def solution_without_collision(self):
        max_steps = 1500
        for _ in range(max_steps):
            # 后者效果更好
            # qd = self.qp_solve_equity_constraint()
            qd = self.qp_solve_target_func()
            # qd = self.using_jacobian()
            # qd = self.qp_solve_target_func_use_slack()
            jacobe = self.robot.jacob0(self.robot.q)
            print("实际输出  (J * qd)  : ", (jacobe[:2, :self.dof] @ qd))
            # print("jacobe: ", self.robot.jacobe(self.robot.q)[:2, :self.dof])
            print("期望输出(expected_v): ", self.get_expected_velocity()[:2])
            J = self.robot.jacob0(self.robot.q)[:2, :]  # 取位置相关的2行
            cond = np.linalg.cond(J)
            print(f"Condition number of J: {cond:.1f}")
            print(f"Rank of J", np.linalg.matrix_rank(J, tol=1e-6))
            print(f"Current q : ", self.robot.q)
            print("\n")

            # update the current angle and current pose
            self.robot.q = self.robot.q + self.dt * qd
            T = self.robot.fkine(self.robot.q)
            # print("T.t: ", T.t[0])
            # print(type(self.current_pose))
            self.current_pose[0] = float(T.t[0])
            self.current_pose[1] = float(T.t[1])

            self.points.append(SE3(T.t[0], T.t[1], 0))
            self.joint_angles.append(self.robot.q)
            # print("Current Position: ", self.current_pose)

        return self.joint_angles
