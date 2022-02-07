import numpy as np
import sympy as sp
from tqdm import tqdm
from scipy.integrate import quad, solve_ivp, odeint
from scipy.optimize import root_scalar, newton, toms748
from scipy.interpolate import splprep, splev
from numpy import pi, sqrt, exp, arcsinh, log as ln

# Constants
def get_const():
    """Constants"""
    c = 2.998e8
    G = 6.674e-11
    hbar = 1.055e-34

    MeV = 1.60218e-19*1e6
    m = 939.57*MeV/c**2

    M0 = 1.988 * 10**30

    u0 = m**4 / (8 * pi**2) * (c**5 / hbar**3) 
    m0 = c**4 / sqrt(4*pi/3 * u0 * G**3) / M0
    r0 = G * m0*M0 / c**2 / 1e3 # (km)
    return u0, m0, r0
u0, m0, r0 = get_const()

# u, p parametrized by fermi momentum x
x = sp.Symbol("x")
u_symb = (2*x**3 + x) * sp.sqrt(1 + x**2) - sp.asinh(x) 
p_symb = 1 / 3 * ((2*x**3 - 3*x) * sp.sqrt(1 + x**2) + 3*sp.asinh(x))
ux = sp.lambdify(x, u_symb, "numpy")
px = sp.lambdify(x, p_symb, "numpy")

# Derivative of p, for solving p(x) - p0
fprime = sp.lambdify(x, p_symb.diff(x).simplify())


def u_fermi(p0, x0=1) -> float:
    """
    Energy density as a funciton of pressure, u = u(p)
    """
    if p0<0: return 0
    f = lambda x: px(x) - p0
    # bracket: x in (0, 1e4) corresponds to u in (0, 2e16)
    x = root_scalar(f, x0=x0, bracket=(0, 1e4)).root
    u0 = ux(x)
    return u0


k = 8/3*(15/8)**(3/5)
def u_fermi_nonrel(p):
    return k*p**(3/5)

# Differential equations to be integrated

def dmdr_general(u, r, y, args):
    """Mass equation in dimensionless units"""
    p, m = y
    return 3 * u(p) * r**2

def dpdr_general(u, r, y, args):
    """TOV equation in dimensionless units, for general equation of state"""
    p, m = y
    if r<1e-10:
        p0 = args
        u0 = u(p0)
        return - r * (p + u(p)) * (3 * p + u0) / (1 - 2*u0*r**2)
    else:
        Dp = - 1 / r**2 * (p + u(p)) * (3 * p * r**3 + m) /  (1 - 2 * m/r)
        return Dp

def stop(r, y, args):
    """Termiation cirerion"""
    p, m = y
    return p
stop.terminal = True # attribute for solve_ivp


N = 151
log_pmin = -6
log_pmax = 6

def sim():
    p0s = 10**np.linspace(log_pmin, log_pmax, N)
    sols = []

    # Specialize to tehe fermi equation of state
    dpdr = lambda r, y, args: dpdr_general(u_fermi, r, y, args)
    dmdr = lambda r, y, args: dmdr_general(u_fermi, r, y, args)
    # Standard ODE form, y'(t) = f(y, t)
    f = lambda r, y, args: (dpdr(r, y, args), dmdr(r, y, args))

    for i, p0 in enumerate(tqdm(p0s)):
        s = solve_ivp(f, (0, 1e3), (p0, 0), args=(p0,), events=stop, max_step=0.001, dense_output=True)
        sols.append(s)
    sols =  np.array(sols)

    np.save("fermi_gas_star/data/sols_neutron", sols)


def sim_nonrel():
    p0s = 10**np.linspace(log_pmin, log_pmax, N)
    sols = []

    # Specialize to tehe fermi equation of state
    dpdr = lambda r, y, args: dpdr_general(u_fermi_nonrel, r, y, args)
    dmdr = lambda r, y, args: dmdr_general(u_fermi_nonrel, r, y, args)
    # Standard ODE form, y'(t) = f(y, t)
    f = lambda r, y, args: (dpdr(r, y, args), dmdr(r, y, args))

    for i, p0 in enumerate(tqdm(p0s)):
        s = solve_ivp(f, (0, 1e3), (p0, 0), args=(p0,), events=stop, max_step=0.001, dense_output=True)
        sols.append(s)
    sols =  np.array(sols)

    np.save("fermi_gas_star/data/sols_neutron_non_rel", sols)


def dpdr_newt(u, r, y, args):
    p, m = y
    p0 = args
    if r < 1e-10:
        return - u(p) * u(p0) * r
    else:
        return - u(p) * m / r**2

def sim_newt():
    p0s = 10**np.linspace(log_pmin, log_pmax, N)
    sols = []

    # Specialize to tehe fermi equation of state
    dpdr = lambda r, y, args: dpdr_newt(u_fermi_nonrel, r, y, args)
    dmdr = lambda r, y, args: dmdr_general(u_fermi_nonrel, r, y, args)
    # Standard ODE form, y'(t) = f(y, t)
    f = lambda r, y, args: (dpdr(r, y, args), dmdr(r, y, args))

    for i, p0 in enumerate(tqdm(p0s)):
        s = solve_ivp(f, (0, 1e3), (p0, 0), args=(p0,), events=stop, max_step=0.01, dense_output=True)
        sols.append(s)
    sols =  np.array(sols)

    np.save("fermi_gas_star/data/sols_neutron_newt", sols)

def sim_newt_rel():
    p0s = 10**np.linspace(log_pmin, log_pmax, N)
    sols = []

    # Specialize to tehe fermi equation of state
    dpdr = lambda r, y, args: dpdr_newt(u_fermi, r, y, args)
    dmdr = lambda r, y, args: dmdr_general(u_fermi, r, y, args)
    # Standard ODE form, y'(t) = f(y, t)
    f = lambda r, y, args: (dpdr(r, y, args), dmdr(r, y, args)) 

    for i, p0 in enumerate(tqdm(p0s)):
        s = solve_ivp(f, (0, 1e3), (p0, 0), args=(p0,), events=stop, max_step=0.001, dense_output=True)
        sols.append(s)
    sols =  np.array(sols)

    np.save("fermi_gas_star/data/sols_neutron_newt_rel", sols)




if __name__ == "__main__":
    # sim()
    # sim_nonrel()
    # sim_newt()
    sim_newt_rel()
