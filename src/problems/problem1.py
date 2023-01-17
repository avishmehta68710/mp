import numpy as np

# Discrete (True) or Continuous Variable (False)
INT_VARS = [False, True]
# Minimization Problem
MAX = False
L_BOUND = [0, 0]
U_BOUND = [1.6, 1]

OPTIMAL = 2


def evaluate(x):
    constraints = np.array([
        1.25 - x[:, 0] ** 2 - x[:, 1],
        np.sum(x, axis=1) - 1.6,
        - x[:, 0],
        x[:, 0] - 1.6,
        - x[:, 1],
        x[:, 1] - 1,
    ])
    print(constraints)
    print(constraints.clip(min=0))
    constraints = np.absolute(constraints.clip(min=0))

    y = 2 * x[:, 0] + x[:, 1]

    return y, np.sum(constraints, axis=0), x[:0]


print(evaluate(np.array([[1.6, 1]])))
