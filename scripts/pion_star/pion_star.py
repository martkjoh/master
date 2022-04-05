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


# sim_EM()

n=201
pcs = 10**np.linspace(-18, 2 , n)

def sim_e():
    u = get_u("pion_star/data/eos_e.npy")
    max_step = 1e-0
    sols = integrate(u, pcs, max_step=max_step, r_max=1e8)
    np.save("pion_star/data/sols_e", sols)

sim_e()

pcs = 10**np.linspace(-5, 5 , n)
def sim_mu():
    u = get_u("pion_star/data/eos_mu.npy")
    max_step = 1e-4
    sols = integrate(u, pcs, max_step=max_step, r_max=1e8)
    np.save("pion_star/data/sols_mu", sols)


sim_mu()


# u = get_u("pion_star/data/eos_mu.npy")
# print(u(1e5))

