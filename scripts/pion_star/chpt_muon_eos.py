import numpy as np
from numpy import pi, sqrt, arcsinh
from scipy.optimize import newton
from matplotlib import pyplot as plt

import sys

sys.path.append(sys.path[0] + "/..")
from constants import get_const_lepton, m_mu
from integrate_tov import get_u


xf = lambda x: sqrt(x**2 - 1)
ul = lambda x: (2*xf(x)**3 + xf(x)) * sqrt(1 + xf(x)**2) - arcsinh(xf(x)) 
pl = lambda x: 1 / 3 * ((2*xf(x)**3 - 3*xf(x)) * sqrt(1 + xf(x)**2) + 3*arcsinh(xf(x)))

p_pi = lambda x: 1/2 * (x - 1/x)**2
u_pi = lambda x: 1/2 * (2 + x**2 - 3/x**2)


u0, ul0, A = get_const_lepton(m_mu)
eq_mu = lambda x, y : A * (y**2 - 1)**(3/2) - x*(1 - 1/x**4)
f_mu = lambda x: newton(lambda y: eq_mu(x, y), np.ones_like(x))

f_mu = lambda x: sqrt( 1 + 1/A**(2/3) * (x * (1-1/x**4))**(2/3)  )



u_mu = lambda x: u_pi(x) + ul0/u0 * ul(f_mu(x))
p_mu = lambda x: p_pi(x) + ul0/u0 * pl(f_mu(x))

m = np.log(10)
N = 1000
def gen_eos_list_mu():
    x = np.logspace(0, m, N)

    ulst = u_mu(x)
    plst = p_mu(x)

    # return ulst, plst

    # Can only interpolate with unique points
    assert len(np.unique(plst)) == len(plst)
    assert len(np.unique(ulst)) == len(ulst)
    np.save("pion_star/data/eos_mu", [x, plst, ulst])


gen_eos_list_mu()


def plot_mus():
    x = np.linspace(1, 2, N)
    plt.plot(x, f_mu(x))
    plt.show()


plot_mus()