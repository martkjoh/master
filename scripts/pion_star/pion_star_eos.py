import numpy as np
from numpy import pi, sqrt




p = lambda x: 1/2 * (x**2 + 1/x**2 - 2)
u = lambda x: 1/2 * (2 + 1/x**2 - 3*x**2)


def gen_eos_list(xrange = (-3, 5), N = 1_000):
    # Define y by x**2 = 1/(1 + y^2), which is "fermi momentu, p_f/p0"
    y = 10**np.linspace(*xrange, N)
    x = 1 / np.sqrt(1 + y**2)

    ulst = u(x)
    plst = p(x)

    # Can only interpolate with unique points
    assert len(np.unique(plst)) == len(plst)
    assert len(np.unique(ulst)) == len(ulst)

    np.save("pion_star/data/eos", [x, plst, ulst])


pEM = lambda x, D: 1/2 * (1/x**2 + (x**2 - 1)/(1 - 2*D*x**2) - 1)
uEM = lambda x, D: 1/2 * (1 + 1/x**2 - (-1 + (3 + 2*D)*x**2 - 2*D*x**4) / (1 - 2*D*x**2))


gen_eos_list()

