from planners import OptimPlanner
from src.BasicParam.BasicArm import Basic3dofArm
from src.BasicParam.Circle import Circle

if __name__ == '__main__':
    circle = Circle(center=(1, -1), radius=1)
    link_lengths = [1.0, 1.0, 1.0]

    arm = Basic3dofArm(link_lengths)
    robot = arm.robot

    planner = (
        OptimPlanner(obj=robot, circle=circle, arm_initial_pose=(0.5, -1)))

    planner.solution_without_collision()
