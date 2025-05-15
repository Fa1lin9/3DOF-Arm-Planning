from roboticstoolbox import DHRobot, RevoluteDH
import numpy as np


# 3-DOF Arm
class Basic3dofArm:
    def __init__(self, linkLengths_):
        if len(linkLengths_) == 0:
            raise ValueError("number of the link must be 3! ")
        for length in linkLengths_:
            if not isinstance(length, (int, float)):
                raise ValueError("link length must be number!")
            if length <= 0:
                raise ValueError("link length must be greater than 0!")

        self.name = "Basic 3DOF Arm"
        self.numJoints = len(linkLengths_)
        self.links = [RevoluteDH(a=_, alpha=0) for _ in linkLengths_]
        self.robot = DHRobot(self.links, name=self.name)

        # Reset the qlim
        # self.robot.qlim = self.robot.qlim * 0.75


if __name__ == "__main__":
    link_lengths = [1.0, 1.0, 1.0]
    arm = Basic3dofArm(link_lengths)
    print(arm.robot.qlim)
    # print(arm.robot.q)
    # print(arm.robot.jacobe(arm.robot.q))
    q = np.array([0.0, 0.0, 0.0])
    print(arm.robot.jacobe(q))
    q = np.deg2rad([10, -30, 20])

    print(arm.robot.jacobe(q))
