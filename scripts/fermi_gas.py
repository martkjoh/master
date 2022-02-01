import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from matplotlib import cm
from scipy.integrate import quad, solve_ivp, odeint
from scipy.optimize import root_scalar, newton, toms748
from scipy.interpolate import splprep, splev
from numpy import pi, sqrt, exp, arcsinh, log as ln

plt.rcParams['mathtext.fontset'] = 'cm'
font = {'family' : 'serif', 
        'size': 20}
plt.rc('font', **font)
plt.rc('lines', lw=2)


c = 2.998e8
G = 6.67e-11
hbar = 1.055e-34

m = 1.67e-27

M0 = 1.98841 * 10**30

u0 = m**4 / (8 * pi**2) * (c**5 / hbar**3) 
m0 = c**4 / sqrt(4*pi/3 * u0 * G**3) / M0
r0 = G * m0*M0 / c**2 / 1e3 # (km)


x = sp.Symbol("x")

u_symb = (2*x**3 + x) * sp.sqrt(1 + x**2) - sp.asinh(x) 
p_symb = 1 / 3 * ((2*x**3 - 3*x) * sp.sqrt(1 + x**2) + 3*sp.asinh(x))

ux = sp.lambdify(x, u_symb, "numpy")
px = sp.lambdify(x, p_symb, "numpy")

fprime = sp.lambdify(x, p_symb.diff(x).simplify())


def u(p0, x0=1) -> float:
    if p0<0: return 0
    f = lambda x: px(x) - p0

    # bracket: x in (0, 1e4) corresponds to u in (0, 2e16)
    x = root_scalar(f, x0=x0, bracket=(0, 1e4)).root
    u0 = ux(x)
    return u0

uv = np.vectorize(u)


def dmdr(r, y, args):
    p, m = y
    return 3 * u(p) * r**2

def dpdr(r, y, args):
    p, m = y
    if r<1e-10:
        p0 = args
        u0 = u(p0)
        return - r * (p + u(p)) * (3 * p + u0) / (1 - 2*u0*r**2)
    else:
        Dp = - 1 / r**2 * (p + u(p)) * (3 * p * r**3 + m) /  (1 - 2 * m/r)
        return Dp


def f(r, y, args):
    return dpdr(r, y, args), dmdr(r, y, args)

def stop(r, y, args):
    p, m = y
    return p
stop.terminal = True

def sim_many():
    N = 100
    log_pmin = -6
    log_pmax = 4
    p0s = 10**np.linspace(log_pmin, log_pmax, N)
    sols = []

    for i, p0 in enumerate(p0s):
        s = solve_ivp(f, (0, 1e3), (p0, 0), args=(p0,), events=stop, max_step=0.001, dense_output=True)
        sols.append(s)
    sols =  np.array(sols)

    np.save("data/sols", sols)


# sim_many()




def load_sols():
    return np.load("data/sols.npy", allow_pickle=True)


def plot_norm_pressure_mass():
    sols = load_sols()
    sols = sols[50:90:2]
    N = len(sols)


    fig1, ax1 = plt.subplots(1, figsize=(10, 7))
    fig2, ax2 = plt.subplots(1, figsize=(10, 7))


    for i, s in enumerate(sols):
        p, m = s.y
        r = s.t
        R = r[-1]
        M = m[-1]
        p0 = p[0]

        c = cm.viridis(i/N)

        ax1.plot(r/R, p/p0, color=c)
        ax1.plot(r/R, p/p0, "--k", lw=1)
        ax2.plot(r/R, m/M, lw=2,  color=c)
        ax2.plot(r/R, m/M, "--k", lw=1)

    fig1.tight_layout()
    fig1.savefig("figurer/pressure.pdf", )
    fig2.tight_layout()
    fig2.savefig("figurer/mass.pdf")


def plot_mass_surface():
    sols = load_sols()
    sols = sols[::]
    N = len(sols)
    m = []
    r = np.linspace(0, 1, 100)
    R = []
    for i, s in enumerate(sols):

        R.append(s.t[-1])
        m.append(s.sol(R[i] * r)[1])  


    m = np.array(m)
    r = np.array(r)
    R = np.array(R)
    r, R = np.meshgrid(r, R)    

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        r, R, m,
        cmap=cm.viridis
        )
    fig.colorbar(surf)
    
    plt.show()


def plot_mass_radius():
    sols = load_sols()
    sols = sols[::]
    N = len(sols)
    M = []
    R = []
    for i, s in enumerate(sols):

        R.append(s.t[-1])
        M.append(s.y[1][-1])  
    R = np.array(R)
    M = np.array(M)

    fig, ax = plt.subplots(figsize=(12, 8))

    Rs = np.linspace(0, 2.5, 100)

    ax.plot(Rs, 4 / 9 * Rs, "k--")
    ax.plot(R*r0, M*m0, "rx")

    n = 1
    P = [(R*r0)[::n], (M*m0)[::n]]

    tck, _ = splprep(P, s=0)
    c = lambda t: splev(t, tck)

    t = np.linspace(0, 1, 1000)
    ax.plot(*c(t), "k", lw=1)

    ax.set_ylim(0, 1)

    ax.set_xlabel(r"$R / \mathrm{km}$")
    ax.set_ylabel(r"$M /  M_\odot$")

    i = np.argmax(M)
    ax.plot(R[i]*r0, M[i]*m0, "ro", ms=15, fillstyle="none")

    R_oppenheimer = [21.1, 13.3, 9.5, 6.8, 3.1]
    M_oppenheimer = [0.4, 0.6, 0.71, 0.64, 0.34]

    ax.plot(R_oppenheimer, M_oppenheimer, "o", ms=10, fillstyle="none")

    fig.tight_layout()
    fig.savefig("figurer/mass_radius.pdf")

plot_norm_pressure_mass()
# plot_mass_surface()
plot_mass_radius()

