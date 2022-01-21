import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from scipy.integrate import quad, solve_ivp
from scipy.optimize import root
from numpy import pi, sqrt, exp, log as ln


x = sp.Symbol("x")
y = sp.sqrt(x**2 + 1)
p_s = y * x * (2 * x + 1) - sp.log(y + x)
u_s = y * x * (2 * x - 3 ) + 3 * sp.log(y + x) 

u_para = sp.lambdify(x, u_s, "numpy")
p_para = sp.lambdify(x, p_s, "numpy")

dpdmu_s = p_s.diff(x).simplify()
dpdmu = sp.lambdify(x, dpdmu_s)

def u(p0, x0= 0.1):
    f = lambda x: p_para(x) - p0
    x = root(f, x0, jac=dpdmu).x[0]
    return u_para(x)

def dmdr(r, y, args):
    p, m = y
    return u(p) * r**2

def dpdr(r, y, args):
    p, m = y
    p0 = args
    u0 = u(p0)
    if r<0.001:
        m = 1/3 * u(0)
        return - r * (p + u(p)) * (p + u0) * (1 - 2 * u0 * r )**(-1)
    else: 
        return - 1 / r**2 * (p + u(p)) * (p * r**3 + m) * (1 - 2 * m / r)**(-1)

def f(r, y, args):
    Dp, Dm = dpdr(r, y, args), dmdr(r, y, args)
    return Dp, Dm

def stop(r, y, args):
    p, m = y
    return p
stop.terminal = True

def sim_many():
    N = 2
    p0s = np.linspace(0.5 , 2, N)

    M = np.zeros(N)
    R = np.zeros(N)

    fig, ax = plt.subplots()

    for i, p0 in enumerate(p0s):
        s = solve_ivp(f, (0, 1e10), (p0, 0), args=(p0,), events=stop)
        rs = s.t
        ps, ms = s.y
        M[i] = ms[-1]
        R[i] = rs[-1]

    n = 15
    plt.plot(rs[:n], ps[:n], "--.")
    plt.show()


sim()