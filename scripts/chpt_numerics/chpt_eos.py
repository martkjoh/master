import numpy as np
from numpy import pi, sqrt
import sys

sys.path.append(sys.path[0] + "/..")
from constants import D


# first approx to alpha as a function of mu_I, analytical result
def alpha_0(mu):
    mu = np.atleast_1d(mu).astype(float)
    morethan_m = mu**2 > np.ones_like(mu)
    a = np.zeros_like(mu)
    a[morethan_m] = np.arccos((1/mu[morethan_m]**2))
    return a


xrange = (0, 5) 
N = 1_000

p = lambda x: 1/2 * (x - 1/x)**2
u = lambda x: 1/2 * (2 + 1/x**2 - 3*x**2) 
nI = lambda x: (1/x**2 - x**2)*x


def gen_eos_list():
    r = (-8, 10)
    y = np.logspace(*r, N-1, dtype=np.longdouble()) 
    y = np.concatenate([[0.,], y])

    x = 1 / np.sqrt(1 + y**2)

    ulst = u(x)
    plst = p(x)
    
    # Can only interpolate with unique points
    assert np.sum(np.diff(plst)>0) == len(plst)-1
    assert np.sum(np.diff(ulst)>0) == len(ulst)-1
    np.save("pion_star/data/eos", [y, plst, ulst])


def u_nr(p):
    """Non-relativistic limit of the fermi gas eos"""
    if p<=0: return 0
    return 2*sqrt(2)*p**(1/2)

u_ur = lambda p: p


# Including EM interactions

pEM = lambda x, D: 1/2 * (1/x**2 +  x**2/(1 - D*x**2) - 2 - D)
uEM = lambda x, D: 1/2 * (
    1/x**2 - x**2*(3 - D*x**2)/(1 - D*x**2)**2 + 2+D
    )


def gen_eos_list_EM():
    r = (-4.4 , 10)
    y = np.logspace(*r, N-1, dtype=np.longdouble()) 
    y = np.concatenate([[0.,], y])


    x = 1 / sqrt(1 + D + y**2)

    ulst = uEM(x, D)
    plst = pEM(x, D)

    # Can only interpolate with unique points
    assert len(np.unique(plst)) == len(plst)
    assert len(np.unique(ulst)) == len(ulst)

    np.save("pion_star/data/eos_EM", [y, plst, ulst])


if __name__=="__main__":
    gen_eos_list()
    gen_eos_list_EM()
