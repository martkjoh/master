import numpy as np
from numpy import pi, sqrt


p = lambda x : 1/2 * x**4/(1 + x**2) / (135/90)**2
u = lambda x : 1/2 * (4*x**2 + x**4) / (1 + x**2) / (135/90)**2


def gen_eos_list(xrange = (-3, 4), N = 200):
    m = 10**np.linspace(*xrange, N)
    ulst = u(m)
    plst = p(m)

    # Can only interpolate with unique points
    assert len(np.unique(plst)) == len(plst)
    assert len(np.unique(ulst)) == len(ulst)

    np.save("pion_star/data/eos", [m, plst, ulst])

gen_eos_list()

