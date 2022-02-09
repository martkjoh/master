import numpy as np

from numpy import pi, sqrt, exp, arcsinh, log as ln
from scipy.integrate import solve_ivp
from scipy.interpolate import splev, splrep
from tqdm import tqdm


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
        return - 1 / r**2 * (p + u(p)) * (3 * p * r**3 + m) /  (1 - 2 * m/r)


def dpdr_newt(u, r, y, args):
    """Newtonian limit of TOV equation for general equation of state"""
    p, m = y
    p0 = args
    if r < 1e-10:
        return - u(p) * u(p0) * r
    else:
        return - u(p) * m / r**2


def get_u(name):
    """ Get equation of state from list of samples """
    # Cubic spline, no smoothing
    _, plst, ulst = np.load(name)
    tck = splrep(plst, ulst, s=0, k=3)

    def u(p):
        if p < plst[0]: return 0
        # High energy limit -- massless particles
        if p > plst[-1]: print("outside samples"); return 3*p
        else: return splev(p, tck)

    return u


def integrate(u, pcs, dense_output=True, max_step=0.001, r_max=1e3, newtonian_limit=False):
    """ Integrate TOV for a list of central pressures pcs"""

    dmdr = lambda r, y, args: dmdr_general(u, r, y, args) 
    if newtonian_limit: dpdr = lambda r, y, args: dpdr_newt(u, r, y, args)
    else: dpdr = lambda r, y, args: dpdr_general(u, r, y, args)
    
    # y'(r) = f(r, y)
    f = lambda r, y, args: (dpdr(r, y, args), dmdr(r, y, args))

    def stop(r, y, args):
        """Termiation cirerion, p(r) = 0"""
        p, m = y
        return p
    stop.terminal = True # attribute for solve_ivp

    sols = [None for _ in pcs] # empty list to keep solutions
    for i, pc in enumerate(tqdm(pcs)):
        sols[i] = solve_ivp(f, (0, r_max), (pc, 0), args=(pc,), events=stop, max_step=max_step, dense_output=dense_output)
    return np.array(sols)

