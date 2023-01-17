import numpy as np
import utils

pmconfig = utils.config.pm
pr = pmconfig.pr
pi = pmconfig.pi
pm = pmconfig.pm


def mutate(offspring, int_vars, l_bound, u_bound):
    new_off = np.zeros_like(offspring)

    for i, off in enumerate(offspring):
        sr = np.random.power(pr)
        si = np.random.power(pi)

        for j, int_var in enumerate(int_vars):
            s = si if int_var else sr

            if np.random.random() < pm:
                xu = u_bound[j]
                xl = l_bound[j]
                xm = off[j]
                t = (xm-xl)/(xu-xl)

                new_off[i, j] = xm - s * \
                    (xm-xl) if t < np.random.random() else xm + s * (xu-xm)
            else:
                new_off[i, j] = off[j]

    return new_off
