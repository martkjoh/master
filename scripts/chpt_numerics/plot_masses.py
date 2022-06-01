import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

sys.path.append(sys.path[0] + "/..")
from spectrum import *


plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)


l = lambda E : lambdify((p, muS, muI, a), lo(E), "numpy")

# Vacuum/pion-condensed masses

mpi0 = lambda muS, muI, a : sqrt(l(E0_sq)(0, muS, muI, a))
mpip = lambda muS, muI, a : sqrt(l(Epip_sq)(0, muS, muI, a+0j))
mpim = lambda muS, muI, a : sqrt(l(Epim_sq)(0, muS, muI, a))
mKp = lambda muS, muI, a : sqrt(l(EKp_sq)(0, muS, muI, a+0j))
mKm = lambda muS, muI, a : sqrt(l(EKm_sq)(0, muS, muI, a+0j))
mK0 = lambda muS, muI, a : sqrt(l(EK0_sq)(0, muS, muI, a+0j))
mK0bar = lambda muS, muI, a : sqrt(l(EK0bar_sq)(0, muS, muI, a+0j))
meta = lambda muS, muI, a : sqrt(l(Eeta_sq)(0, muS, muI, a+0j))

# EM-masses

mpi0EM = lambda muS, muI, a : sqrt(l(E0_sq_EM)(0, muS, muI, a))
mpipEM = lambda muS, muI, a : sqrt(l(Epip_sq_EM)(0, muS, muI, a+0j))
mpimEM = lambda muS, muI, a : sqrt(l(Epim_sq_EM)(0, muS, muI, a))

mKpEM = lambda muS, muI, a : sqrt(l(EKp_sq_EM)(0, muS, muI, a+0j))
mKmEM = lambda muS, muI, a : sqrt(l(EKm_sq_EM)(0, muS, muI, a+0j))
mK0EM = lambda muS, muI, a : sqrt(l(EK0_sq_EM)(0, muS, muI, a+0j))
mK0barEM = lambda muS, muI, a : sqrt(l(EK0bar_sq_EM)(0, muS, muI, a+0j))


def plot_meson_masses():
    fig, ax = plt.subplots(2 ,figsize=(12, 12), sharex=True)
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
    ax[0].set_ylabel("$m/m_{\\pi}$")
    ax[1].set_ylabel("$m/m_{\\pi}$")

    ax[0].legend()
    ax[1].legend()

    fig.savefig("figurer/masses_mesons.pdf", bbox_inches="tight")


def plot_meson_em_masses():
    fig, ax = plt.subplots(2 ,figsize=(12, 12), sharex=True)
    mu_list = np.linspace(0, 2.5, 400)
    alpha_list = alpha_EM(mu_list)

    ax[1].plot(mu_list, mpi0(0, mu_list, alpha_list), "-", color="tab:blue", label="$\\pi^{0}$")
    ax[1].plot(mu_list, mpipEM(0, mu_list, alpha_list), "r-.", label="$\\pi^{+}$")
    ax[1].plot(mu_list, mpimEM(0, mu_list, alpha_list).real, "k--",  label="$\\pi^{-}$")

    muS_n = 0
    ax[0].plot(mu_list, mKpEM(muS_n, mu_list, alpha_list), "k--", label="$K^+$")
    ax[0].plot(mu_list, mKmEM(muS_n, mu_list, alpha_list), "k-.",  label="$K^-$")
    ax[0].plot(mu_list, mK0(muS_n, mu_list, alpha_list), "r--", label="$K^0$")
    ax[0].plot(mu_list, mK0bar(muS_n, mu_list, alpha_list), "r-.",  label="$\\bar K^0$")
    ax[0].plot(mu_list, meta(muS_n, mu_list, alpha_list), "-", color="tab:blue", label="$\\eta$")

    ax[1].set_xlabel("$\\mu_I/m_{\\pi}$")
    ax[0].set_ylabel("$m/m_{\\pi}$")
    ax[1].set_ylabel("$m/m_{\\pi}$")

    ax[0].legend()
    ax[1].legend()

    fig.savefig("figurer/masses_mesons_EM.pdf", bbox_inches="tight")


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
plot_charged_kaon_masses2()
plot_meson_em_masses()

