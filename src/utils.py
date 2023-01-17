import json
import numpy as np
from types import SimpleNamespace

with open('config.json') as jsfile:
    config = json.load(
        jsfile, object_hook=lambda i: SimpleNamespace(**i))


def generate(n, int_vars, l_bound, u_bound):
    gen = []

    for _ in range(0, n):
        x = np.zeros(len(int_vars))
        for j, int_var in enumerate(int_vars):
            x[j] = np.random.randint(l_bound[j], u_bound[j] + 1) if int_var else \
                l_bound[j] + (u_bound[j] - l_bound[j]) * np.random.rand()
        gen.append(x)

    return np.array(gen)


def evaluate(population, f_eval, max=False):
    fitness = np.zeros(population.shape[0])
    f_fworst = np.min if max else np.max

    score, constraint = f_eval(population)

    feasible_idx = np.where(constraint == 0)
    feasible = score[feasible_idx]
    fworst = f_fworst(feasible) \
        if feasible.size > 0 else 0

    fitness = fworst - constraint if max else fworst + constraint
    fitness[feasible_idx] = score[feasible_idx]

    return fitness, feasible.size > 0
