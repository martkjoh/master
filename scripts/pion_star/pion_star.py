import numpy as np
import sys

sys.path.append(sys.path[0] + "/..")
from integrate_tov import get_u, integrate


n = 201
pcs = 10**np.linspace(-6, 6, n)


def sim():
    u = get_u("pion_star/data/eos.npy")
    max_step = 1e-3
    sols = integrate(u, pcs, max_step=max_step)
    np.save("pion_star/data/sols", sols)



sim()

pcs = 10**np.linspace(-6, 6, n)

def sim_EM():
    u = get_u("pion_star/data/eos_EM.npy")
    max_step = 1e-3
    sols = integrate(u, pcs, max_step=max_step)
    np.save("pion_star/data/sols_EM", sols)


sim_EM()
