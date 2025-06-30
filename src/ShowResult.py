import matplotlib.pyplot as plt
import matplotlib.patches as patches
from planners import *
from BasicParam import *

if __name__ == '__main__':
    # circle = Circle(center=(0, 0.5), radius=2)
    # circle = Circle(center=(-1, 0.5), radius=1)
    circle = Circle(center=(0.5, -0.5), radius=2)
    init_pose = (1, 1)
    link_lengths = [1.0, 1.0, 1.0]

    arm = Basic3dofArm(link_lengths)
    robot = arm.robot

    planner_flag = 1
    if planner_flag == 1:
        analyticalPlanner = (
            AnalyticalPlanner(obj=robot, circle=circle, arm_initial_pose=init_pose, n=50))
        joint_angles = analyticalPlanner.solution_without_collision()
    elif planner_flag == 2:
        optimalPlanner = OptimPlanner(obj=robot, circle=circle, arm_initial_pose=init_pose )
        joint_angles = optimalPlanner.solution_without_collision()
    elif planner_flag == 3:
        samplingPlanner = SamplingPlanner(obj=robot, circle=circle, arm_initial_pose=init_pose, )
        joint_angles = samplingPlanner.solution_without_collision()
        print(joint_angles)

    # 创建图形
    fig, ax = plt.subplots()
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    line, = ax.plot([], [], 'o-', lw=4)

    # draw a circle
    circle = patches.Circle(circle.center, circle.radius, fill=False, color='gray', linestyle='--')
    ax.add_patch(circle)

    # 初始化末端执行器轨迹
    end_effector_trajectory = []

    def update(q):
        # 正向运动学，计算机械臂各关节的位置
        T = robot.fkine_all(q)
        x = [t.t[0] for t in T]
        y = [t.t[1] for t in T]

        # 末端执行器的位置
        end_effector_position = T[-1].t[:2]
        end_effector_trajectory.append(end_effector_position)

        # 更新机械臂的位置
        line.set_data(x, y)

        # 绘制末端执行器的轨迹
        end_effector_x = [pos[0] for pos in end_effector_trajectory]
        end_effector_y = [pos[1] for pos in end_effector_trajectory]
        ax.plot(end_effector_x, end_effector_y, 'r-', lw=1)  # 末端执行器轨迹为红色
        return line


    for q in joint_angles:
        update(q)
        plt.pause(0.05)

    plt.show()
