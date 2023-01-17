import numpy as np

# Discrete (True) or Continuous Variable (False)
INT_VARS = [True, True, True, True, True, True, True]
# Minimization Problem
MAX = False
L_BOUND = [0, 0, 0, 0, 0, 0, 0]
U_BOUND = [4, 4, 4, 2, 2, 2, 6]

OPTIMAL = 14


def evaluate(x):
    constraints = np.array([
        x[:, 0] + x[:, 1] + x[:, 2] - 6,
        x[:, 3] + x[:, 4] + 6 * x[:, 5] - 8,
        x[:, 0] * x[:, 5] + x[:, 1] + 3 * x[:, 4] - 7,
        4 * x[:, 1] * x[:, 6] + 3 * x[:, 3] * x[:, 4] - 25,
        3 * x[:, 0] + 2 * x[:, 2] + x[:, 4] - 7,
        20 - 3 * x[:, 0] * x[:, 2] - 6 * x[:, 3] - 4 * x[:, 4],
        15 - 4 * x[:, 0] - 2 * x[:, 2] - x[:, 5] * x[:, 6],
        x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 6],
        4 - x[:, 0], 4 - x[:, 1], 4 - x[:, 2],
        2 - x[:, 3], 2 - x[:, 4], 2 - x[:, 5],
        6 - x[:, 6]
    ])
    constraints = np.absolute(constraints.clip(max=0))

    y = x[:, 0] * x[:, 6] + 3 * x[:, 1] * \
        x[:, 5] + x[:, 2] * x[:, 4] + 7 * x[:, 3]

    return y, np.sum(constraints, axis=0)
