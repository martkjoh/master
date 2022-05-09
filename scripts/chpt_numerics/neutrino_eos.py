import numpy as np
from numpy import pi, sqrt, arcsinh
from scipy.optimize import newton
from matplotlib import pyplot as plt
import sys

sys.path.append(sys.path[0] + "/..")
from constants import get_const_lepton, f_pi, m_e, m_mu, m_pi
from integrate_tov import get_u

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)



def const():
    _, ue0, Ae = get_const_lepton(m_e)
    _, umu0, Amu = get_const_lepton(m_mu)
    a = m_e/m_mu
    u0 = f_pi**2*m_pi**2
    return a, Ae, Amu, ue0, umu0, u0

a, Ae, Amu, ue0, umu0, u0 = const()


def xi(x):
    l = x>1
    x0 = np.ones_like(x)
    x0[l] = x[l]
    return x0

def eq(x, y): # x = mu_I/m_\pi, y = mu_e / m_e
    more = a*y > 1
    assert np.all(np.diff(more)>=0) # more should only go from 0 to 1
    y2 = np.ones_like(y)/a
    y2[more] = y[more]
    return xi(x)*(1 - 1/xi(x)**4) - Ae * (xi(y)**2 - 1)**(3/2) - Amu * ((a*y2)**2 - 1)**(3/2)

def mu_e(x):
    _, ue0, Ae = get_const_lepton(m_e)
    return sqrt(1 + 1/Ae**(2/3) * ( xi(x)*(1-1/xi(x)**4) )**(2/3))

def x_e(x_I):
    x_e0 = mu_e(x_I)

    f = lambda x_e: eq(x_I, x_e)
    x, conv, _ = newton(f, x_e0, full_output=True)
    assert np.all(conv)
    return x

# x = mu_\ell
def xf(x):
    l = x>1
    x0 = np.ones_like(x)
    x0[l] = sqrt(x[l]**2 - 1)
    return x0

xf = lambda x: sqrt(x**2 - 1)
def pl(x):
    p0 = np.zeros_like(x)
    m = x>1
    p0[m] = 1 / 3 * ((2*xf(x[m])**3 - 3*xf(x[m])) * x[m] + 3*arcsinh(xf(x[m])))
    return p0

def ul(x):
    u0 = np.zeros_like(x)
    m = x>1
    u0[m] = (2*xf(x[m])**3 + xf(x[m])) * x[m] - arcsinh(xf(x[m]))
    return u0

xnu = lambda x: x + m_e/m_pi * x_e(x)
pnu = lambda x: xnu(x)**4 / (24 * pi**2)
unu = lambda x: xnu(x)**4 / (8 * pi**2)

# x = mu_I
p_pi = lambda  x: 1/2 * (xi(x) - 1/xi(x))**2
u_pi = lambda x: 1/2 * (2 + xi(x)**2 - 3/xi(x)**2)


pe = lambda x: ue0/u0*pl(x_e(x))
ue = lambda x: ue0/u0*ul(x_e(x))
pmu = lambda x: umu0/u0*pl(a*x_e(x))
umu = lambda x: umu0/u0*ul(a*x_e(x))
 
p = lambda x: p_pi(x) + pe(x) + pmu(x) + 2*pnu(x)
u = lambda x: u_pi(x) + ue(x) + umu(x) + 2*unu(x)


def plot_mu():
    N = 1000
    x = np.linspace(1, 1.1, N)
    y = x_e(x)

    fig, ax = plt.subplots(figsize=(9, 5)) 
    ax.plot(x, y, label="$\mu_e(\mu_I)$")
    ax.plot(x, mu_e(x), "k--", label="$\mu_e'(\mu_I)$")

    ax.set_xlabel("$\\mu_I/m_\\pi$")
    ax.set_ylabel("$\mu_e/m_e$")


    dy = a* np.diff(y)/np.diff(x)
    ax2 = plt.twinx(ax)
    dmu = "$\\frac{\\mathrm{d} \\mu_e}{\\mathrm{d} \\mu_I}$"
    ax2.plot(x[:-1]+np.diff(x), dy, "k-.", lw=1, label=dmu)
    ax2.set_ylim(-2.05, 45)
    ax2.grid(False)
    dmu = "${\\mathrm{d} \\mu_e}/{\\mathrm{d} \\mu_I}$"
    ax2.set_ylabel(dmu)

    fig.legend(loc=(0.65, 0.4))
    
    fig.savefig("figurer/neutrino_mu.pdf", bbox_inches="tight")


pmin = 2*(1+m_e/m_pi) / (24*pi**2)
def plot_eos():
    x = np.linspace(0, 2.2, 1000)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p(x), 3*p(x), "k--", label="$u = 3 p$")
    ax.plot(p(x), u(x), label="$u(p)$")
    ax.set_xlabel("$p/u_{0}$")
    ax.set_ylabel("$u/u_{0}$")
    plt.legend()

    fig.savefig("figurer/neutrino_eos.pdf", bbox_inches="tight")


def plot_eos2():
    x = np.linspace(0, 1.01, 1000)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p(x), 3*p(x), "k--", label="$u = 3 p$")
    ax.plot(p(x), u(x), label="$u(p)$")
    ax.plot(pmin, 3*pmin, "kx", label="$p_\\mathrm{min}$")
    plt.legend()
    ax.set_xlabel("$p/u_{0}$")
    ax.set_ylabel("$u/u_{0}$")

    fig.savefig("figurer/neutrino_eos2.pdf", bbox_inches="tight")
    

def save_eos():
    x = np.concatenate([np.linspace(0, 1, 100), 1+np.logspace(-14, 2 , 1000)])
    ulst = u(x)
    plst = p(x)
    
    # Can only interpolate with unique points
    assert len(np.unique(plst)) == len(plst)
    assert len(np.unique(ulst)) == len(ulst)
    assert np.sum(np.diff(plst)<0) == 0
    assert np.sum(np.diff(ulst)<0) == 0
    np.save("pion_star/data/eos_neutrino", [x, plst, ulst])


plot_mu()

plot_eos()
plot_eos2()

save_eos()

