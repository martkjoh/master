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


Epim_sq, Epip_sq = get_E(m1_sq, m2_sq, m12)
EKm_sq, EKp_sq = get_E(m4_sq, m4_sq, m45)
EK0bar_sq, EK0_sq = get_E(m6_sq, m6_sq, m67)


# first approx to alpha as a function of mu_I, analytical result
def alpha_0(mu):
    mu = np.atleast_1d(mu).astype(float)
    morethan_m = mu**2 > np.ones_like(mu)
    a = np.zeros_like(mu)
    a[morethan_m] = arccos((1/mu[morethan_m]**2))
    return a


l = lambda E : lambdify((p, muS, muI, a), lo(E),"numpy") 

mpi0 = lambda muS, muI, a : sqrt(l(E0_sq)(0, muS, muI, a))
mpip = lambda muS, muI, a : sqrt(l(Epip_sq)(0, muS, muI, a+0j))
mpim = lambda muS, muI, a : sqrt(l(Epim_sq)(0, muS, muI, a))

mKp = lambda muS, muI, a : sqrt(l(EKp_sq)(0, muS, muI, a+0j))
mKm = lambda muS, muI, a : sqrt(l(EKm_sq)(0, muS, muI, a+0j))
mK0 = lambda muS, muI, a : sqrt(l(EK0_sq)(0, muS, muI, a+0j))
mK0bar = lambda muS, muI, a : sqrt(l(EK0bar_sq)(0, muS, muI, a+0j))
meta = lambda muS, muI, a : sqrt(l(Eeta_sq)(0, muS, muI, a+0j))


def plot_meson_masses():
    fig, ax = plt.subplots(2 ,figsize=(10, 14), sharex=True)
    mu_list = np.linspace(0, 2.5, 400)
    alpha_list = alpha_0(mu_list)

    ax[1].plot(mu_list, mpi0(0, mu_list, alpha_list), "-", color="tab:blue", label="$\\pi^{0}$")
    ax[1].plot(mu_list, mpip(0, mu_list, alpha_list), "r-.", label="$\\pi^{+}$")
    ax[1].plot(mu_list, mpim(0, mu_list, alpha_list).real, "k--",  label="$\\pi^{-}$")

    muS_n = 0
    ax[0].plot(mu_list, mKp(muS_n, mu_list, alpha_list), "k--", label="$K^+$")
    ax[0].plot(mu_list, mKm(muS_n, mu_list, alpha_list), "k-.",  label="$K^-$")
    ax[0].plot(mu_list, mK0(muS_n, mu_list, alpha_list), "r--", label="$K^0$")
    ax[0].plot(mu_list, mK0bar(muS_n, mu_list, alpha_list), "r-.",  label="$\\bar K^0$")
    ax[0].plot(mu_list, meta(muS_n, mu_list, alpha_list), "-", color="tab:blue", label="$\\eta$")

    ax[1].set_xlabel("$\\mu_I/m_\\pi$")
    ax[0].set_ylabel("$m/m_\\pi$")
    ax[0].set_ylabel("$m/m_\\pi$")

    ax[0].legend()
    ax[1].legend()
    fig.savefig("figurer/masses_mesons.pdf", bbox_inches="tight")
    

def plot_charged_kaon_masses():
    fig, ax = plt.subplots(figsize=(12, 8))
    mu_list = np.linspace(0, 2.5, 400)
    alpha_list = alpha_0(mu_list)
    muS_n = 0 
    


    ax.set_xlabel("$\\mu_I/m_\\pi$")
    ax.set_ylabel("$m/m_\pi$")
    ax.set_ylim(2.9, 4.18)

    plt.legend(loc="lower right")
    fig.savefig("figurer/masses_kaons.pdf", bbox_inches="tight")



def plot_charged_kaon_masses2():
    fig, ax = plt.subplots(figsize=(12, 5))
    muI_n = 0.5
    a = alpha_0(muI_n)
    mKpm = sqrt( (m_pi**2 + m_S**2 - Dm**2)/2 )/m_pi
    muS = np.linspace(0, mKpm, 500)
    b = mKpm- 1/2*muI_n
    ax.plot([b, b], [0, 10], "-k", alpha=0.4)

    ax.plot(muS, mKp(muS, muI_n, a), "k--", label="$K^+$")
    ax.plot(muS, mKm(muS, muI_n, a), "k-.",  label="$K^-$")
    ax.plot(muS, mK0(muS, muI_n, a), "r--", label="$K^0$")
    ax.plot(muS, mK0bar(muS, muI_n, a), "r-.",  label="$\\bar K^0$")
    ax.set_title("$\mu_I = %.2f" %muI_n + "\\, m_\\pi$")

    ax.set_xlabel("$\\mu_S/m_\\pi$")
    ax.set_ylabel("$m/m_\pi$")

    plt.legend(loc="upper left")
    ax.set_ylim(-0.2, 8.2)
    fig.savefig("figurer/masses_kaons2.pdf", bbox_inches="tight")



plot_meson_masses() 
plot_charged_kaon_masses()
plot_charged_kaon_masses2()
