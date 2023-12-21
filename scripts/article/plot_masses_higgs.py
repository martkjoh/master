import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

sys.path.append(sys.path[0] + "/..")
from constants_phys import f_pi, m_Kpm, m_K0, e, m_pipm
from constants_LO import Dm, Dm_EM, m_pi0


m_eta = np.sqrt((2*m_Kpm**2 + 2*m_K0**2 - m_pi0**2) / 3)

f_pi, m_pi0, Dm, m_Kpm, m_K0, Dm_EM, m_eta = np.array([f_pi, m_pi0, Dm, m_Kpm, m_K0, Dm_EM, m_eta]) / m_pipm
plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)


def mu_Ieffsq(mu):
    return mu**2 - Dm_EM**2

def m1sq(sgn, mu):
    if mu_Ieffsq(mu) < m_pi0**2:
        return ( m_Kpm**2 + m_K0**2 + m_pi0**2 + sgn * np.sqrt( (m_Kpm**2 + m_K0**2 - 2* m_pi0**2)**2+ 3*Dm**4  ) )/3
    mt = 2 / 3 * (m_K0**2 + m_Kpm**2 - m_pi0**2)
    term1 = 3 * mu_Ieffsq(mu) * (mt + mu_Ieffsq(mu)) + m_pi0**4 
    term2 = 12 * Dm**4 * mu_Ieffsq(mu)**2 + (3*mu_Ieffsq(mu) * (mt - mu_Ieffsq(mu)) + m_pi0**4)**2
    return ( term1 + sgn* np.sqrt(term2) )/ (6 * mu_Ieffsq(mu))

def m2sq(sgn, mu):
    if mu_Ieffsq(mu) < m_pi0**2:
        return (m_K0 +  sgn * mu/2)**2
    term1 = 2*m_pi0**2 * mu**2 * (m_pi0**2 - mu_Ieffsq(mu)) + mu_Ieffsq(mu)**2 * (4*m_K0**2 + mu**2)
    term2 = 4*m_pi0**4 * mu**2 *(
        mu**2 * m_pi0**4 + mu_Ieffsq(mu) * (4 * m_K0**2*mu_Ieffsq(mu) + mu**2 * (mu_Ieffsq(mu) - 2*m_pi0**2))
        )
    return ( term1 + sgn * np.sqrt(term2) )/ (4 * mu_Ieffsq(mu)**2)

def m3sq(sgn, mu):
    if mu_Ieffsq(mu) < m_pi0**2:
        return (np.sqrt(m_Kpm**2 + Dm_EM**2) + sgn * mu/2)**2
    term1 = 2*m_pi0**2  * (mu**2*m_pi0**2 - mu_Ieffsq(mu)*(mu_Ieffsq(mu) - Dm_EM**2)) + mu_Ieffsq(mu)**2 * (4*m_Kpm**2 + mu**2)
    term2 = 4*m_pi0**4 * mu**2 *(
        mu**2 * m_pi0**4 + mu_Ieffsq(mu) * (4 * m_Kpm**2*mu_Ieffsq(mu) + mu**2 * (mu_Ieffsq(mu) - 2*m_pi0**2) + 4*Dm_EM**2*m_pi0**2)
        )
    return ( term1 + sgn * np.sqrt(term2) )/ (4 * mu_Ieffsq(mu)**2)

def mm(mu):
    if mu_Ieffsq(mu) < m_pi0**2:
        return (np.sqrt(m_pi0**2 + Dm_EM**2 ) + mu)**2
    return (mu_Ieffsq(mu)**3 + 3*mu_Ieffsq(mu)*m_pi0**4 + 4*Dm_EM**2*m_pi0**4) / mu_Ieffsq(mu)**2

def mp(mu):
    if mu_Ieffsq(mu) < m_pi0**2:
        return (np.sqrt(m_pi0**2 + Dm_EM**2) - mu)**2
    return e**2 * f_pi**2 * (1 - m_pi0**2 / mu_Ieffsq(mu))

def mA(mu):
    if mu_Ieffsq(mu) < m_pi0**2:
        return 0
    return e**2 * f_pi**2 * (1 - m_pi0**2 / mu_Ieffsq(mu))


mus = np.linspace(0, 5, 2000)

m1p = np.array([np.sqrt(m1sq(+1, mu)) for mu in mus])
m1m = np.array([np.sqrt(m1sq(-1, mu)) for mu in mus])
m2p = np.array([np.sqrt(m2sq(+1, mu)) for mu in mus])
m2m = np.array([np.sqrt(m2sq(-1, mu)) for mu in mus])
m3p = np.array([np.sqrt(m3sq(+1, mu)) for mu in mus])
m3m = np.array([np.sqrt(m3sq(-1, mu)) for mu in mus])
mm = np.array([np.sqrt(mm(mu)) for mu in mus])
mp = np.array([np.sqrt(mp(mu)) for mu in mus])
mA = np.array([np.sqrt(mA(mu)) for mu in mus])

fig, ax = plt.subplots(2, sharex=True, figsize=(12,12))
ax[0].plot(mus, m1p, '-', color='tab:blue', lw=2, label='$\\eta$')
ax[0].plot(mus, m2p, 'r-.', lw=2, label='$\\bar K^0$')
ax[0].plot(mus, m2m, 'r--', lw=2, label='$K^0$')
ax[0].plot(mus, m3p, 'k-.', lw=2, label='$K^-$')
ax[0].plot(mus, m3m, 'k--', lw=2, label='$K^+$')
ax[0].set_ylim(3., 4.1)
ax[0].legend()

ax[1].plot(mus, mA, 'g', lw=2,  label='$A_\\mu$')
ax[1].plot(mus, mm, 'r--', lw=2, label='$\\pi^-$')
ax[1].plot(mus, mp, 'k--', lw=2, label='$\\pi^+$')
ax[1].plot(mus, m1m, '-', color='tab:blue', lw=2, label='$\\pi_0$')
ax[1].set_ylim(-.1, 2.5)
ax[1].set_xlim(0, 2.5)
ax[1].legend()

ax[1].set_xlabel("$\\mu_I/m_{\\pi^\pm}$")
ax[0].set_ylabel("$m/m_{\\pi^\pm}$")
ax[1].set_ylabel("$m/m_{\\pi^\pm}$")

fig.savefig("article/figurer/masses_higgs.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(mus, m1p, '-', color='tab:blue', lw=2, label='$\\eta$')
ax.plot(mus, m2p, 'r-.', lw=2, label='$\\bar K^0$')
ax.plot(mus, m2m, 'r--', lw=2, label='$K^-$')
ax.plot(mus, m3p, 'k-.', lw=2, label='$K^0$')
ax.plot(mus, m3m, 'k--', lw=2, label='$K^+$')
ax.plot(mus, mm, 'r--', lw=2, label='$\\pi^-$')
ax.plot(mus, m1m, '-', color='tab:blue', lw=2, label='$\\eta$')

ax.set_xlabel("$\\mu_I/m_{\\pi^\pm}$")
ax.set_ylabel("$m/m_{\\pi^\pm}$")
ax.set_xlim(3.75, 4.25)
ax.set_ylim(3.75, 4.25)
fig.savefig("article/figurer/masses_higgs3.pdf", bbox_inches="tight")
