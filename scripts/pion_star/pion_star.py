import numpy as np
import sys

sys.path.append(sys.path[0] + "/..")
from integrate_tov import get_u, integrate


n = 201
pcs = 10**np.linspace(-6, 6, n)


def sim():
    u = get_u("pion_star/data/eos.npy")
    sols = integrate(u, pcs, max_step=2e-3)
    np.save("pion_star/data/sols", sols)

sim()