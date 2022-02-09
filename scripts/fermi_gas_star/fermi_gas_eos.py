import numpy as np
import sympy as sp
from tqdm import tqdm
from scipy.integrate import quad, solve_ivp, odeint
from scipy.optimize import root_scalar, newton, toms748
from scipy.interpolate import splprep, splev
from numpy import pi, sqrt, exp, arcsinh, log as ln

import sys
sys.path.append(sys.path[0] + "/..")

from constants import get_const_fermi_gas

u0, m0, r0 = get_const_fermi_gas()

# u, p parametrized by fermi momentum x
x = sp.Symbol("x")
u_symb = (2*x**3 + x) * sp.sqrt(1 + x**2) - sp.asinh(x) 
p_symb = 1 / 3 * ((2*x**3 - 3*x) * sp.sqrt(1 + x**2) + 3*sp.asinh(x))
ux = sp.lambdify(x, u_symb, "numpy")
px = sp.lambdify(x, p_symb, "numpy")


def u_fermi_root(p0, x0=1) -> float:
    """
    Energy density as a funciton of pressure, u = u(p), by solving for root
    """
    if p0<0: return 0
    f = lambda x: px(x) - p0
    # bracket: x in (0, 1e4) corresponds to u in (0, 2e16)
    x = root_scalar(f, x0=x0, bracket=(0, 1e4)).root
    u0 = ux(x)
    return u0


# x = 10(-3.5 -- 2) corresponds to u = (8.4e-11 -- 2e8)
def gen_eos_list(xrange = (-3.5, 2), N = 201):
    """Generate eos samples"""
    x = 10**np.linspace(*xrange, N)
    ulst = ux(x)
    plst = px(x)

    # Can only interpolate with unique points
    assert len(np.unique(plst)) == len(plst)
    assert len(np.unique(ulst)) == len(ulst)

    np.save("fermi_gas_star/data/eos", [x, plst, ulst])


k = 8/3*(15/8)**(3/5)
def u_fermi_nonrel(p):
    """Non-relativistic limit of the fermi gas eos"""
    if p<=0: return 0
    return k*p**(3/5)


if __name__=="__main__":
    gen_eos_list()
