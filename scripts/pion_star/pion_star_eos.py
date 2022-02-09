import numpy as np
import sympy as sp
from tqdm import tqdm
from scipy.integrate import quad, solve_ivp, odeint
from scipy.optimize import root_scalar, newton, toms748
from scipy.interpolate import splprep, splev
from numpy import pi, sqrt, exp, arcsinh, log as ln



u = lambda y : 1 - 1/y**2 + 3/2 * (y**2 - 1/y**2)
p = lambda y : 1/y**2 - 1 + 1/2 * (y**2 - 1/y**2)

def gen_eos_list(mrange = (0, 5), N = 200):
    m = 10**np.linspace(*mrange, N)
    ulst = u(m)
    plst = p(m)

    # Can only interpolate with unique points
    assert len(np.unique(plst)) == len(plst)
    assert len(np.unique(ulst)) == len(ulst)

    np.save("pion_star/data/eos", [m, plst, ulst])

gen_eos_list()


def sim_pion_star():
    N = 100
    log_pmin = 0
    log_pmax = 4
    _, plst, ulst = get_data("pion_star/data/eos.npy")
    u = get_u(plst, ulst)

    p0s = 10**np.linspace(log_pmin, log_pmax, N)
    sols = [None for _ in p0s]

    # Specialize to tehe fermi equation of state

    # Standard ODE form, y'(t) = f(y, t)
    f = lambda r, y, args: (dpdr(r, y, args), dmdr(r, y, args))

    for i, p0 in enumerate(tqdm(p0s)):
        s = solve_ivp(f, (0, 1e2), (p0, 0), args=(p0,), events=stop, max_step=0.01, dense_output=True)
        sols[i] = s
    sols = np.array(sols)

    np.save("pion_star/data/sols_tree", sols)

