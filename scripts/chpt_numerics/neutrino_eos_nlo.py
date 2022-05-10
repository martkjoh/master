import numpy as np
from numpy import pi, sqrt, arcsinh
from scipy.optimize import newton
from matplotlib import pyplot as plt
import sys

sys.path.append(sys.path[0] + "/..")
from constants import get_const_lepton, f_pi, m_e, m_mu, m_pi
from integrate_tov import get_u
from chpt_numerics.free_energy_nlo import get_p_u

lattice = True
if lattice:
    from constants_lattice import get_const_lepton, f_pi, m_e, m_mu, m_pi
    l = "lattice"
else:
    from constants import get_const_lepton, f_pi, m_e, m_mu, m_pi
    l = ""


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
p_pi, u_pi = get_p_u(l)


pe = lambda x: ue0/u0*pl(x_e(x))
ue = lambda x: ue0/u0*ul(x_e(x))
pmu = lambda x: umu0/u0*pl(a*x_e(x))
umu = lambda x: umu0/u0*ul(a*x_e(x))
 
p = lambda x: p_pi(x) + pe(x) + pmu(x) + 2*pnu(x)
u = lambda x: u_pi(x) + ue(x) + umu(x) + 2*unu(x)



pmin = 2*(1+m_e/m_pi) / (24*pi**2)
def plot_eos():
    x = np.linspace(0, 6.5, 1000)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p(x), 3*p(x), "k--", label="$u = 3 p$")
    ax.plot(p(x), u(x), label="$u(p)$")
    ax.set_xlabel("$p/u_{0}$")
    ax.set_ylabel("$u/u_{0}$")
    plt.legend()
    plt.show()

    fig.savefig("figurer/neutrino_nlo_eos.pdf", bbox_inches="tight")
    

def save_eos():
    x = np.concatenate([np.linspace(0, 1, 100), 1+np.logspace(-14, np.log10(5.5), 1000)])
    ulst = u(x)
    plst = p(x)
    
    # Can only interpolate with unique points
    assert len(np.unique(plst)) == len(plst)
    assert len(np.unique(ulst)) == len(ulst)
    assert np.sum(np.diff(plst)<0) == 0
    assert np.sum(np.diff(ulst)<0) == 0
    np.save("pion_star/data/eos_neutrino_nlo"+l, [x, plst, ulst])



# plot_eos()
save_eos()

