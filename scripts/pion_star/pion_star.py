import numpy as np
import sys

from chpt_eos import u_nr
# from chpt_lepton_eos import u
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


def sim_EM(r=(-6, 6)):
    pcs = 10**np.linspace(r, n)
    u = get_u("pion_star/data/eos_EM.npy")
    max_step = 1e-3
    sols = integrate(u, pcs, max_step=max_step)
    np.save("pion_star/data/sols_EM", sols)


def sim_e(r=(-18, 2), max_step=1e-1, info=False):
    pcs = np.logspace(*r , n)
    u = get_u("pion_star/data/eos_e.npy")
    sols = integrate(u, pcs, max_step=max_step, r_max=1e8, info=info)
    np.save("pion_star/data/sols_e", sols)


def sim_mu(r=(-12, 4), max_step=1e-1, info=False):
    pcs = np.logspace(*r , n)
    u = get_u("pion_star/data/eos_mu.npy")
    sols = integrate(u, pcs, max_step=max_step, r_max=1e8, info=info)
    np.save("pion_star/data/sols_mu", sols)

# sim()
# sim_non_rel()
# sim_newt()
# sim_newt_non_rel()

# sim_EM()

# sim_e(max_step=1e0)
sim_mu(max_step=1e-2)



