import numpy as np
import sympy as sp
from tqdm import tqdm
from scipy.integrate import quad, solve_ivp, odeint
from scipy.optimize import root_scalar, newton, toms748
from scipy.interpolate import splprep, splev
from numpy import pi, sqrt, exp, arcsinh, log as ln

from constants import get_const_fermi_gas

# Constants
u0, m0, r0 = get_const_fermi_gas()

N = 151
log_pmin = -6
log_pmax = 6
pcs = 10**np.linspace(log_pmin, log_pmax, N)


def sim_neutron():
    u = get_u("fermi_gas_star/data/eos.npy")
    sols = integrate(u, pcs)
    np.save("fermi_gas_star/data/sols_neutron", sols)

def sim_nonrel():
    u = u_fermi_nonrel
    sols = integrate(u, pcs)
    np.save("fermi_gas_star/data/sols_neutron_non_rel", sols)


def sim_newt():
    u = get_u("fermi_gas_star/data/eos.npy")
    sols = integrate(u, pcs, newtonian_limit=True)
    np.save("fermi_gas_star/data/sols_neutron_newt", sols)


def sim_newt_rel():
    u = u_fermi_nonrel
    sols = integrate(u, pcs, newtonian_limit=True)
    np.save("fermi_gas_star/data/sols_neutron_newt_rel", sols)



if __name__ == "__main__":
    sim()
    sim_nonrel()
    sim_newt()
    sim_newt_rel()
