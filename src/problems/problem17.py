import numpy as np

# Discrete (True) or Continuous Variable (False)
INT_VARS = [True for _ in range(0, 100)]
# Minimization Problem
MAX = True
L_BOUND = [0 for _ in range(0, 100)]
U_BOUND = [99 for _ in range(0, 100)]

OPTIMAL = 303062089

POP_SIZE = 3


def evaluate(x):

    a = np.array([50, 150, 100, 92, 55, 12, 11, 10, 8, 3,
                  114, 90, 87, 91, 58, 16, 19, 22, 21, 32,
                  53, 56, 118, 192, 52, 204, 250, 295, 82, 30,
                  29, -2, 9, 94, 15, 17, -15, -2, 1, 3,
                  52, 57, -1, 12, 21, 6, 7, -1, 1, 1,
                  119, 82, 75, 18, 16, 12, 6, 7, 3, 6,
                  12, 13, 18, 7, 3, 19, 22, 3, 12, 9,
                  18, 19, 12, 8, 5, 2, 16, 17, 11, 12,
                  9, 12, 11, 14, 16, 3, 9, 10, 3, 1,
                  12, 3, 12, -2, -1, 6, 7, 4, 1, 2])

    p = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  2, 2, 2, 1, 3, 2, 1, 1, 1, 4,
                  1, 2, 2, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 2, 1, 1, 1, 1, 1, 1])

    c1 = np.array([x[:, i] - 7500 for i in range(0, 100)])

    c21 = c22 = 0
    for i in range(0, 50):
        c21 = c21 + 10 * x[:, 0]
    for i in range(0, 100):
        c22 = c22 + x[:, 0]
    c2 = c21 + c22 - 42000

    c31 = np.array([-x[:, i] for i in range(0, 100)])
    c32 = np.array([x[:, i] - 99 for i in range(0, 100)])

    constraints = np.vstack((c1, c2, c31, c32))
    constraints = np.absolute(constraints.clip(min=0))

    y = np.sum(a * x ** p, axis=1)
    return y, np.sum(constraints, axis=0)
