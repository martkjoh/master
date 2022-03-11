import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import sys


from sympy import lambdify
from numpy import arccos, sqrt
from matplotlib import cm

sys.path.append(sys.path[0] + "/..")
from constants import f_pi_MeV

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)


lo = lambda x: x.subs(m, 1.).subs(f, f_pi_MeV)


# Everything is done in units of m_pi
m, f = sp.symbols("m, f")
# This is usefull to write rational numbers
one = sp.S.One
a, mu, p = sp.symbols("a, mu, p", real=True)


# Mass-parameters
m1_sq = m**2 * sp.cos(a) - mu**2 * sp.cos(2 * a)
m2_sq = m**2 * sp.cos(a) - mu**2 * sp.cos(a)**2
m3_sq = m**2 * sp.cos(a) + mu**2 * sp.sin(a)**2

m12 = 2 * mu * sp.cos(a)

E0_sq = p**2 + m2_sq
M_sq = (m1_sq + m2_sq + m12**2)
Ep_sq = p**2 + 1 / 2 * M_sq + 1/2 * sp.sqrt(4 * p**2 * m12**2 + M_sq**2 - 4*m1_sq*m2_sq)
Em_sq = p**2 + 1 / 2 * M_sq - 1/2 * sp.sqrt(4 * p**2 * m12**2 + M_sq**2 - 4*m1_sq*m2_sq)

# Tree-level masses. Should equal E(p=0)
m0_sq = m3_sq
mp_sq = 1 / 2 * M_sq + 1/2 * sp.sqrt(M_sq**2 - 4*m1_sq*m2_sq)
mm_sq = 1 / 2 * M_sq - 1/2 * sp.sqrt(M_sq**2 - 4*m1_sq*m2_sq)


# first approx to alpha as a function of mu_I, analytical result
def alpha_0(mu):
    morethan_m = mu**2 > np.ones_like(mu)
    a = np.zeros_like(mu)
    a[morethan_m] = arccos((1/mu[morethan_m]**2))
    return a


def plot_masses():
    fig, ax = plt.subplots(figsize=(8, 5))
    mu_list = np.linspace(0, 2.5, 100)
    alpha_list = alpha_0(mu_list)
    m0 = lambda x, y : sqrt(lambdify((mu, a), lo(m0_sq), "numpy")(x, y))
    mp = lambda x, y : sqrt(lambdify((mu, a), lo(mp_sq), "numpy")(x, y))
    mm = lambda x, y : sqrt(lambdify((mu, a), lo(mm_sq), "numpy")(x+0j, y+0j))

    assert not np.sum(np.abs(mm(mu_list, alpha_list).imag) > 1e-6)
    
    ax.plot(mu_list, m0(mu_list, alpha_list), "-", color="tab:blue", label=r"$m_{0}$")
    ax.plot(mu_list, mp(mu_list, alpha_list), "r-.", label=r"$m_{+}$")
    ax.plot(mu_list, mm(mu_list, alpha_list).real, "k--",  label=r"$m_{-}$")

    ax.set_xlabel(r"$\mu_I/m_\pi$")
    ax.set_ylabel(r"$m/m_\pi$")

    ax.legend()
    fig.savefig("figurer/leading_order_masses.pdf", bbox_inches="tight")

plot_masses()
