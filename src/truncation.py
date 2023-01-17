import numpy as np


def truncate(offspring, int_vars):
    new_off = np.copy(offspring)

    for i in range(len(offspring)):
        for j, int_var in enumerate(int_vars):
            if int_var:
                new_off[i, j] = np.ceil(offspring[i, j]) if np.random.random(
                ) > 0.5 else np.floor(offspring[i, j])
    return new_off
