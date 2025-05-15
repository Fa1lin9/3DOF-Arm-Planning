import numpy as np
from qpsolvers import solve_qp

# 定义二次规划问题的参数
P = np.array([[2, 0], [0, 2]], dtype=float)  # 二次项系数矩阵，确保是 float 类型
q = np.array([1, 1], dtype=float)  # 线性项系数

# 约束条件: Ax <= b
A = np.array([[-1, 0], [0, -1]], dtype=float)  # 约束矩阵，确保是 float 类型
b = np.array([-1, -1], dtype=float)  # 约束右侧的常数项，确保是 float 类型

# 求解二次规划问题
x = solve_qp(P, q, A, b, solver="scs")

# 输出结果
print("Optimal solution:", x)
