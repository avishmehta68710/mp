import numpy as np
import problems.problem1 as p1


def select(population, score, size, max=False):

    selection = np.zeros((size, population.shape[1]))

    for i in range(0, size):
        idx = np.random.choice(population.shape[0], 2, replace=False)
        top_idx = np.argmax(score[idx]) if max else np.argmin(score[idx])
        selection[i] = population[idx[top_idx]]
    return selection
