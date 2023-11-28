import numpy as np
from numpy import sqrt, pi, log as ln
from scipy.optimize import fsolve



c = 2.998e8
hbar = 1.055e-34
kB = 1.380e-23
G = 6.674e-11

MeV = 1.60218e-19*1e6
e = 0.3028

M0 = 1.988 * 10**30

# pion decay constant
f_pi = 92.07

# Masses
m_pi0 = 134.98
m_pipm = 139.57
m_Kpm = 493.68
m_K0 = 497.61
m_eta = 547.86

m_e = 0.5110
m_mu = 105.7

m_N = 939.57*MeV/c**2

m_rho = 770
f_rho = 154


def eq(x):
    mp, mk, dm, dmEM = x
    return (
        m_pi0**2 -   ( 2*mk**2 + mp**2 + dm**2 - np.sqrt( (2*mk**2 - 2*mp**2 + dm**2 )**2 + 3*dm**4 ))/3 ,
        m_pipm**2 - (mp**2 + dmEM**2),
        m_K0**2 -   (mk**2 + dm**2),
        m_Kpm**2 -  (mk**2 + dmEM**2)
    )


sol = fsolve(eq, (m_pi0, m_K0, 70, 35))
(m_pi0, m_Kpm0, dm, dm_EM) = sol
Dm, Dm_EM = (71.60, 35.09) # Rounding

m_u = (m_pi0**2 - Dm**2) / 2
m_d = (m_pi0**2 + Dm**2) / 2
m_S = m_Kpm0**2 - m_u
m_K00 = sqrt(m_d + m_S)

C = f_pi**2/(2*e**2) * dm_EM**2
# Curech = 3 / (32*pi**2) * m_rho**2*f_rho**2 * ln(f_rho**2 /(f_rho**2 - f_pi**2))
# print(C)


print(sqrt(m_u + m_d))
print(sqrt(m_u + m_S))
