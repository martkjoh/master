import numpy as np
import sys

sys.path.append(sys.path[0] + "/..")
from fermi_gas_eos import u_fermi_nonrel
from integrate_tov import get_u, integrate

N = 201
log_pmin = -6
log_pmax = 6
pcs = 10**np.linspace(log_pmin, log_pmax, N)


def sim():
    u = get_u("fermi_gas_star/data/eos.npy")
    sols = integrate(u, pcs, dense_output=True)
    np.save("fermi_gas_star/data/sols_neutron", sols)

def sim_non_rel():
    u = u_fermi_nonrel
    sols = integrate(u, pcs)
    np.save("fermi_gas_star/data/sols_neutron_non_rel", sols)


def sim_newt():
    u = get_u("fermi_gas_star/data/eos.npy")
    sols = integrate(u, pcs, newtonian_limit=True)
    np.save("fermi_gas_star/data/sols_neutron_newt", sols)


def sim_newt_non_rel():
    u = u_fermi_nonrel
    sols = integrate(u, pcs, newtonian_limit=True)
    np.save("fermi_gas_star/data/sols_neutron_newt_non_rel", sols)



if __name__ == "__main__":
    # Run to generate all data

    sim()
    sim_non_rel()
    sim_newt() 
    sim_newt_non_rel()

    # smol
    sols = np.load("fermi_gas_star/data/sols_neutron.npy", allow_pickle=True)
    np.save("fermi_gas_star/data/sols_neutron_small", sols[::5])
