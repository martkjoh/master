import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import sys


from sympy import lambdify
from numpy import arccos, sqrt
from matplotlib import cm

sys.path.append(sys.path[0] + "/..")
from constants import f_pi, m_S, m_pi, Dm, m_Kpm

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)


lo = lambda x: x.subs(m, 1.).subs(f, f_pi/m_pi).subs(mS, m_S/m_pi).subs(dm, Dm/m_pi)

print((m_Kpm-m_pi/2)/ m_pi)
# Everything is done in units of m_pi
m, mS, f = sp.symbols("m, m_S, f")
# This is usefull to write rational numbers
one = sp.S.One
a = sp.symbols("alpha")
muI = sp.symbols("mu_I")
muS = sp.symbols("mu_S")
dm = sp.symbols("Delta_m")
p = sp.symbols("p", real=True)

# Mass-parameters
m1_sq = m**2 * sp.cos(a) - muI**2 * sp.cos(2 * a)
m2_sq = m**2 * sp.cos(a) - muI**2 * sp.cos(a)**2
m3_sq = m**2 * sp.cos(a) + muI**2 * sp.sin(a)**2
m8_sq = 1/3*(m**2*sp.cos(a) + 2*mS**2)

m_p_sq = one/2 * (m**2*sp.cos(a) + mS**2 + dm**2)
m_m_sq = one/2 * (m**2*sp.cos(a) + mS**2 - dm**2)
m_mu_p_sq = one/4 * muI**2*sp.cos(2*a) + muI*muS*sp.cos(a) + muS**2
m_mu_m_sq = one/4 * muI**2*sp.cos(2*a) - muI*muS*sp.cos(a) + muS**2

m4_sq = m_m_sq - m_mu_p_sq
m6_sq = m_p_sq - m_mu_m_sq

m12 = 2 * muI * sp.cos(a)
m45 = muI*sp.cos(a) + 2*muS
m67 = muI*sp.cos(a) - 2*muS

E0_sq = p**2 + m3_sq
Eeta_sq = p**2 + m8_sq

def get_E(m1_sq, m2_sq, m12, p=p):
    M_sq = (m1_sq + m2_sq + m12**2)
    Ep_sq = \
        p**2 \
        + 1 / 2 * M_sq \
        + 1/2 * sp.sqrt(4 * p**2 * m12**2 + M_sq**2 - 4*m1_sq*m2_sq)
    Em_sq = \
        p**2 + 1 / 2 * M_sq\
        - 1/2 * sp.sqrt(4 * p**2 * m12**2 + M_sq**2 - 4*m1_sq*m2_sq)
    return Ep_sq, Em_sq

Epip_sq, Epim_sq = get_E(m1_sq, m2_sq, m12)
EKp_sq, EKm_sq = get_E(m4_sq, m4_sq, m45)
EK0_sq, EK0bar_sq = get_E(m6_sq, m6_sq, m67)

# first approx to alpha as a function of mu_I, analytical result
def alpha_0(mu):
    mu = np.atleast_1d(mu)
    morethan_m = mu**2 > np.ones_like(mu)
    a = np.zeros_like(mu)
    a[morethan_m] = arccos((1/mu[morethan_m]**2))
    return a


l = lambda E : lambdify((p, muS, muI, a), lo(E),"numpy") 

mpi0 = lambda muS, muI, a : sqrt(l(E0_sq)(0, muS, muI, a))
mpip = lambda muS, muI, a : sqrt(l(Epip_sq)(0, muS, muI, a))
mpim = lambda muS, muI, a : sqrt(l(Epim_sq)(0, muS, muI, a+0j))

mKp = lambda muS, muI, a : sqrt(l(EKp_sq)(0, muS, muI, a+0j))
mKm = lambda muS, muI, a : sqrt(l(EKm_sq)(0, muS, muI, a+0j))
mK0 = lambda muS, muI, a : sqrt(l(EK0_sq)(0, muS, muI, a+0j))
mK0bar = lambda muS, muI, a : sqrt(l(EK0bar_sq)(0, muS, muI, a+0j))
meta = lambda muS, muI, a : sqrt(l(Eeta_sq)(0, muS, muI, a+0j))


def plot_pion_masses():
    fig, ax = plt.subplots(figsize=(8, 5))
    mu_list = np.linspace(0, 2.5, 100)
    alpha_list = alpha_0(mu_list)

    ax.plot(mu_list, mpi0(0, mu_list, alpha_list), "-", color="tab:blue", label="$m_{0}$")
    ax.plot(mu_list, mpip(0, mu_list, alpha_list), "r-.", label="$m_{+}$")
    ax.plot(mu_list, mpim(0, mu_list, alpha_list).real, "k--",  label="$m_{-}$")

    ax.set_xlabel("$\\mu_I/m_\\pi$")
    ax.set_ylabel("$m/m_\\pi$")

    ax.legend()
    plt.show()


def plot_charged_kaon_masses(muS_n):
    fig, ax = plt.subplots(figsize=(8, 5))
    mu_list = np.linspace(0, 2.5, 100)
    alpha_list = alpha_0(mu_list)
    
    ax.plot(mu_list, mKp(muS_n, mu_list, alpha_list), "k--", label="$K^+$")
    ax.plot(mu_list, mKm(muS_n, mu_list, alpha_list), "k-.",  label="$K^-$")
    ax.plot(mu_list, mK0(muS_n, mu_list, alpha_list), "r--", label="$K^0$")
    ax.plot(mu_list, mK0bar(muS_n, mu_list, alpha_list), "r-.",  label="$\\bar K^0$")
    ax.plot(mu_list, meta(muS_n, mu_list, alpha_list), ":", color="tab:blue", label="$\\eta$")

    ax.set_xlabel("$\\mu_I/m_\\pi$")
    ax.set_ylabel("$m/m_\pi$")

    plt.legend()
    plt.show()

def plot_charged_kaon_masses2():
    fig, ax = plt.subplots(figsize=(8, 5))
    muI_n = 1
    a = 0
    muS = np.linspace(0, 5, 500)

    y1 = mKp(muS, muI_n, a)
    y2 = mKm(muS, muI_n, a)
    y3 = mK0(muS, muI_n, a)
    y4 = mK0bar(muS, muI_n, a)
    print(y2)
    (y1, y2, y3, y4) = [y[np.abs(y.imag)<1e-6] for y in (y1, y2, y3, y4)]
    ax.plot(muS, y1, "k--", label="$K^+$")
    ax.plot(muS, y2, "k-.",  label="$K^-$")
    ax.plot(muS, y3, "r--", label="$K^0$")
    ax.plot(muS, y4, "r-.",  label="$\\bar K^0$")
    # ax.plot(muS, meta(muS, muI_n, a), ":", color="tab:blue", label="$\\eta$")

    ax.set_xlabel("$\\mu_S/m_\\pi$")
    ax.set_ylabel("$m/m_\pi$")

    plt.legend()
    plt.show()




# plot_pion_masses() 
plot_charged_kaon_masses(0)
plot_charged_kaon_masses(1)
plot_charged_kaon_masses2()

