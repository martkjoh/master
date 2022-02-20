import numpy as np
from numpy import pi, sqrt


xrange = (0, 5) 
N = 1_000

p = lambda x: 1/2 * (x**2 + 1/x**2 - 2)
u = lambda x: 1/2 * (2 + 1/x**2 - 3*x**2)


def gen_eos_list():
    xrange = (-3, 10) 
    y = 10**np.linspace(*xrange, N)
    x = 1 / np.sqrt(1 + y**2)

    ulst = u(x)
    plst = p(x)

    # Can only interpolate with unique points
    assert len(np.unique(plst)) == len(plst)
    assert len(np.unique(ulst)) == len(ulst)
    print(plst)
    np.save("pion_star/data/eos", [x, plst, ulst])


D = 0.06916

pEM = lambda x, D: 1/2 * (1/x**2 +  x**2/(1 - 2*D*x**2) - 2*(1+D))
uEM = lambda x, D: 1/2 * (
    1/x**2 - x**2*(3 - 2*D*x**2)/(1 - 2*D*x**2)**2 + 2*(1+D)
    )


def gen_eos_list_EM():
    y = np.logspace(-3, 5, N)
    x = 1 / sqrt(1 + 2*D + y**2)

    ulst = uEM(x, D)
    plst = pEM(x, D)

    # Can only interpolate with unique points
    assert len(np.unique(plst)) == len(plst)
    assert len(np.unique(ulst)) == len(ulst)

    np.save("pion_star/data/eos_EM", [x, plst, ulst])




# gen_eos_list()
gen_eos_list_EM()
