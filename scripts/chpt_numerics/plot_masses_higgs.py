import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

sys.path.append(sys.path[0] + "/..")
from constants import f_pi, m_S, m_pi, Dm, m_Kpm, m_K0, Dm_EM
m_eta = np.sqrt((2*m_Kpm**2 + 2*m_K0**2 - m_pi**2) / 3)

f_pi, m_S, m_pi, Dm, m_Kpm, m_K0, Dm_EM, m_eta = np.array([f_pi, m_S, m_pi, Dm, m_Kpm, m_K0, Dm_EM, m_eta]) / m_pi

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)


def mu_Ieffsq(mu):
    return mu**2 - Dm_EM**2

def m1sq(sgn, mu):
    if mu_Ieffsq(mu) < m_pi**2:
        return ( m_Kpm**2 + m_K0**2 + m_pi**2 + sgn * np.sqrt( (m_Kpm**2 + m_K0**2 - 2* m_pi**2)**2+ 3*Dm**4  ) )/3
    mt = 2 / 3 * (m_K0**2 + m_Kpm**2 - m_pi**2)
    term1 = 3 * mu_Ieffsq(mu) * (mt + mu_Ieffsq(mu)) + m_pi**4 
    term2 = 12 * Dm**4 * mu_Ieffsq(mu)**2 + (3*mu_Ieffsq(mu) * (mt - mu_Ieffsq(mu)) + m_pi**4)**2
    return ( term1 + sgn* np.sqrt(term2) )/ (6 * mu_Ieffsq(mu))

def m2sq(sgn, mu):
    if mu_Ieffsq(mu) < m_pi**2:
        return (m_K0 + sgn * mu/2)**2
    term1 = 2*m_pi**2 * mu**2 * (m_pi**2 - mu_Ieffsq(mu)) + mu_Ieffsq(mu)**2 * (4*m_K0**2 + mu**2)
    term2 = 4*m_pi**4 * mu**2 *(
        mu**2 * m_pi**4 + mu_Ieffsq(mu) * (4 * m_K0**2*mu_Ieffsq(mu) + mu**2 * (mu_Ieffsq(mu) - 2*m_pi**2))
        )
    return ( term1 + sgn * np.sqrt(term2) )/ (4 * mu_Ieffsq(mu)**2)

def m3sq(sgn, mu):
    if mu_Ieffsq(mu) < m_pi**2:
        return (m_Kpm + sgn * mu/2)**2
    term1 = 2*m_pi**2  * (mu**2*m_pi**2 - mu_Ieffsq(mu)*(mu_Ieffsq(mu) - Dm_EM**2)) + mu_Ieffsq(mu)**2 * (4*m_Kpm**2 + mu**2)
    term2 = 4*m_pi**4 * mu**2 *(
        mu**2 * m_pi**4 + mu_Ieffsq(mu) * (4 * m_Kpm**2*mu_Ieffsq(mu) + mu**2 * (mu_Ieffsq(mu) - 2*m_pi**2) + 4*Dm_EM**2*m_pi**2)
        )
    return ( term1 + sgn * np.sqrt(term2) )/ (4 * mu_Ieffsq(mu)**2)

def mm(mu):
    if mu_Ieffsq(mu) < m_pi**2:
        return (m_pi + mu)**2
    return (mu_Ieffsq(mu)**3 + m_pi**2*(3*mu_Ieffsq(mu)**2 + 4*Dm_EM**4)) / mu_Ieffsq(mu)**2

fig, ax = plt.subplots()

mus = np.linspace(0, 4.3, 2000)

m1p = np.array([np.sqrt(m1sq(+1, mu)) for mu in mus])
m1m = np.array([np.sqrt(m1sq(-1, mu)) for mu in mus])
m2p = np.array([np.sqrt(m2sq(+1, mu)) for mu in mus])
m2m = np.array([np.sqrt(m2sq(-1, mu)) for mu in mus])
m3p = np.array([np.sqrt(m3sq(+1, mu)) for mu in mus])
m3m = np.array([np.sqrt(m3sq(-1, mu)) for mu in mus])
mm = np.array([np.sqrt(mm(mu)) for mu in mus])

ax.plot(mus, m1p)
ax.plot(mus, m1m)
ax.plot(mus, m2p)
ax.plot(mus, m2m)
ax.plot(mus, m3p)
ax.plot(mus, m3m)
ax.plot(mus, mm)


plt.show()

