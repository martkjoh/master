import numpy as np
import sys

from pion_star_eos import u_nr
sys.path.append(sys.path[0] + "/..")
from integrate_tov import get_u, integrate


n = 201
pcs = 10**np.linspace(-6, 6, n)


def sim():
    u = get_u("pion_star/data/eos.npy")
    max_step = 1e-3
    sols = integrate(u, pcs, max_step=max_step)
    np.save("pion_star/data/sols", sols)


def sim_non_rel():
    u = u_nr
    sols = integrate(u, pcs)
    np.save("pion_star/data/sols_non_rel", sols)


def sim_newt():
    u = get_u("pion_star/data/eos.npy")
    sols = integrate(u, pcs, newtonian_limit=True)
    np.save("pion_star/data/sols_newt", sols)


def sim_newt_non_rel():
    u = u_nr
    sols = integrate(u, pcs, newtonian_limit=True)
    np.save("pion_star/data/sols_newt_non_rel", sols)



# sim()
# sim_non_rel()
# sim_newt()
# sim_newt_non_rel()

pcs = 10**np.linspace(-6, 6, n)

def sim_EM():
    u = get_u("pion_star/data/eos_EM.npy")
    max_step = 1e-3
    sols = integrate(u, pcs, max_step=max_step)
    np.save("pion_star/data/sols_EM", sols)


sim_EM()
