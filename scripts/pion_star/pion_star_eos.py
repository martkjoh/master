import numpy as np
import sympy as sp
from tqdm import tqdm
from scipy.integrate import quad, solve_ivp, odeint
from scipy.optimize import root_scalar, newton, toms748
from scipy.interpolate import splprep, splev
from numpy import pi, sqrt, exp, arcsinh, log as ln


p = lambda x : 1/2 * x**4/(1 + x**2)
u = lambda x : 1/2 * (4*x**2 + x**4) / (1 + x**2)


def gen_eos_list(xrange = (-3, 4), N = 200):
    m = 10**np.linspace(*xrange, N)
    ulst = u(m)
    plst = p(m)

    # Can only interpolate with unique points
    assert len(np.unique(plst)) == len(plst)
    assert len(np.unique(ulst)) == len(ulst)

    np.save("pion_star/data/eos", [m, plst, ulst])

gen_eos_list()

