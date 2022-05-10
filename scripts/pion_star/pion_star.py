import numpy as np
import sys
from numpy import pi
from tqdm import tqdm
# from chpt_lepton_eos import u
sys.path.append(sys.path[0] + "/..")
from chpt_numerics.chpt_eos import u_nr
from integrate_tov import get_u, integrate
from constants import m_e, m_pi
from chpt_numerics.make_table import find_p_u_c

n = 201

def sim():
    pcs = 10**np.linspace(-6, 6, n)
    u = get_u("pion_star/data/eos.npy")
    max_step = 5e-4
    sols = integrate(u, pcs, max_step=max_step, dense_output=True)
    np.save("pion_star/data/sols", sols)


def sim_nlo():
    u = get_u("pion_star/data/eos_nlo.npy")
    max_step = 5e-4
    n = 201
    pcs = 10**np.linspace(-6, np.log10(30), n)
    sols = integrate(u, pcs, max_step=max_step)
    np.save("pion_star/data/sols_nlo", sols)


def sim_nlo_lattice():
    u = get_u("pion_star/data/eos_nlolattice.npy")
    max_step = 5e-4
    n = 201
    pcs = 10**np.linspace(-6, np.log10(30), n)
    sols = integrate(u, pcs, max_step=max_step)
    np.save("pion_star/data/sols_nlolattice", sols)



def sim_non_rel():
    pcs = 10**np.linspace(-6, 6, n)
    u = u_nr
    sols = integrate(u, pcs)
    np.save("pion_star/data/sols_non_rel", sols)


def sim_newt():
    pcs = 10**np.linspace(-6, 6, n)
    u = get_u("pion_star/data/eos.npy")
    sols = integrate(u, pcs, newtonian_limit=True)
    np.save("pion_star/data/sols_newt", sols)


def sim_newt_non_rel():
    pcs = 10**np.linspace(-6, 6, n)
    u = u_nr
    sols = integrate(u, pcs, newtonian_limit=True)
    np.save("pion_star/data/sols_newt_non_rel", sols)


def sim_EM(r=(-6, 6)):
    pcs = 10**np.linspace(*r, n)
    u = get_u("pion_star/data/eos_EM.npy")
    max_step = 5e-4
    sols = integrate(u, pcs, max_step=max_step, dense_output=True)
    np.save("pion_star/data/sols_EM", sols)


def sim_e(r=(-18, 2), max_step=1e-1, info=False):
    pcs = np.logspace(*r , n)
    u = get_u("pion_star/data/eos_e.npy")
    sols = integrate(u, pcs, max_step=max_step, r_max=1e8, info=info, pmin=1e-40)
    np.save("pion_star/data/sols_e", sols)


def sim_mu(r=(-12, 4), max_step=1e-1, info=False):
    pcs = np.logspace(*r , n)
    u = get_u("pion_star/data/eos_mu.npy")
    sols = integrate(u, pcs, max_step=max_step, r_max=1e8, info=info, pmin=1e-40)
    np.save("pion_star/data/sols_mu", sols)


pmin = (1+m_e/m_pi) / (12*pi**2)

def sim_neut(r=(np.log(pmin), 0), max_step=1e-3, info=False):
    pcs = np.logspace(np.log10(pmin*1.01), np.log10(pmin*2) , 50)
    pcs = np.concatenate([pcs, np.logspace(np.log10(2*pmin), np.log10(pmin*5) , 50)])
    pcs = np.concatenate([pcs, np.logspace(np.log10(5*pmin), 2.1, 101)])

    u = get_u("pion_star/data/eos_neutrino.npy")
    sols = integrate(u, pcs, max_step=max_step, r_max=1e8, info=info, pmin=pmin, dense_output=False)
    np.save("pion_star/data/sols_neutrino", sols)


def sim_light(max_step=1e-3, info=False):
    n=201
    pmins = [0.1, 10**(-1.5), (1+m_e/m_pi) / (12*pi**2), 10**(-2.5), 0.001]
    for pmin in pmins:
        pcs = np.logspace(np.log10(pmin*1.01), np.log10(pmin*2) , 50)
        pcs = np.concatenate([pcs, np.logspace(np.log10(2*pmin), np.log10(pmin*5) , 50)])
        pcs = np.concatenate([pcs, np.logspace(np.log10(5*pmin), 2.1, 101)])        
        
        u = lambda p: 3*p
        sols = integrate(u, pcs, max_step=max_step, r_max=1e8, info=info, pmin=pmin, dense_output=False)
        np.save("pion_star/data/sols_light_%.2e"%pmin, sols)


def sim_max_pure():
    names = ["", "_nlo", "_EM", "_e", "_mu", "_neutrino"]
    max_steps = [1e-3, 1e-3, 1e-3, 1e-1, 1e-1, 1e-3]
    pmins = [1e-40, 1e-40, 1e-40, 1e-40, 1e-40, (1+m_e/m_pi) / (12*pi**2)]
    for i, name in enumerate(tqdm(names)):
        R, uc_max, pc_max, j = find_p_u_c(name)
        u = get_u("pion_star/data/eos"+name+".npy")
        sol = integrate(u, [pc_max,], max_step=max_steps[i], dense_output=True, progress=False, pmin=pmins[i])[0]
        np.save("pion_star/data/max"+name, sol)



# print("pure")
# sim()
# sim_non_rel()
# sim_newt()
# sim_newt_non_rel()

# print("nlo")
# sim_nlo()
# sim_nlo_lattice()


# print("EM")
# sim_EM()

# print("l")
# sim_e(max_step=1e0)
# sim_mu(max_step=1e-2)

# print("nu")
# sim_neut()

sim_light()

# sim_max_pure()
