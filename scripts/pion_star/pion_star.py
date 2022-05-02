import numpy as np
import sys
from numpy import pi

from chpt_eos import u_nr
# from chpt_lepton_eos import u
sys.path.append(sys.path[0] + "/..")
from integrate_tov import get_u, integrate
from constants import m_e, m_pi

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


pmin = 2*(1+m_e/m_pi) / (24*pi**2)
def sim_neut(r=(np.log(pmin), 0), max_step=1e-3, info=False):
    pcs = np.logspace(np.log10(pmin*1.01), np.log10(pmin*2) , 50)
    pcs = np.concatenate([pcs, np.logspace(np.log10(2*pmin), np.log10(pmin*5) , 50)])
    pcs = np.concatenate([pcs, np.logspace(np.log10(5*pmin), 2.1, 101)])

    u = get_u("pion_star/data/eos_neutrino.npy")
    sols = integrate(u, pcs, max_step=max_step, r_max=1e8, info=info, pmin=pmin, dense_output=False)
    np.save("pion_star/data/sols_neutrino", sols)


pmin = 0.0085
n=51
pmins = [0.1, 0.01, 0.001]
def sim_light(max_step=1e-3, info=False):
    for pmin in pmins:
        r=(np.log10(pmin), 1)
        pcs = np.logspace(*r, n)
        u = lambda p: 3*p
        sols = integrate(u, pcs, max_step=max_step, r_max=1e8, info=info, pmin=pmin, dense_output=False)
        np.save("pion_star/data/sols_light_%.2e"%pmin, sols)



# sim()
# sim_non_rel()
# sim_newt()
# sim_newt_non_rel()

# sim_EM()

# sim_e(max_step=1e0)
# sim_mu(max_step=1e-2)
# sim_neut()
sim_light()

