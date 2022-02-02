import numpy as np
import sympy as sp
from tqdm import tqdm
from scipy.integrate import quad, solve_ivp, odeint
from scipy.optimize import root_scalar, newton, toms748
from scipy.interpolate import splprep, splev
from numpy import pi, sqrt, exp, arcsinh, log as ln

# Constants
def get_const(m = 1.67e-27):
    # Constants
    c = 2.998e8
    G = 6.67e-11
    hbar = 1.055e-34

    M0 = 1.98841 * 10**30

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


def u(p0, x0=1) -> float:
    """
    Energy density as a funciton of pressure, u = u(p)
    """
    if p0<0: return 0
    f = lambda x: px(x) - p0
    # bracket: x in (0, 1e4) corresponds to u in (0, 2e16)
    x = root_scalar(f, x0=x0, bracket=(0, 1e4)).root
    u0 = ux(x)
    return u0


# Differential equations to be integrated

def dmdr(r, y, args):
    """Mass equation in dimensionless units"""
    p, m = y
    return 3 * u(p) * r**2

def dpdr(r, y, args):
    """TOV equation in dimensionless units"""
    p, m = y
    if r<1e-10:
        p0 = args
        u0 = u(p0)
        return - r * (p + u(p)) * (3 * p + u0) / (1 - 2*u0*r**2)
    else:
        Dp = - 1 / r**2 * (p + u(p)) * (3 * p * r**3 + m) /  (1 - 2 * m/r)
        return Dp


def f(r, y, args):
    """y'(r) = f(y, r)"""
    return dpdr(r, y, args), dmdr(r, y, args)

def stop(r, y, args):
    """Termiation cirerion"""
    p, m = y
    return p
stop.terminal = True # attribute for solve_ivp


def sim_many():
    N = 30
    log_pmin = -4
    log_pmax = 2
    p0s = 10**np.linspace(log_pmin, log_pmax, N)
    sols = []

    for i, p0 in enumerate(tqdm(p0s)):
        s = solve_ivp(f, (0, 1e3), (p0, 0), args=(p0,), events=stop, max_step=0.01, dense_output=True)
        sols.append(s)
    sols =  np.array(sols)

    np.save("fermi_gas_star/data/sols_electron", sols)


if __name__ == "__main__":
    sim_many()

