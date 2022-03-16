import numpy as np
from numpy import sqrt, pi


c = 2.998e8
G = 6.674e-11
hbar = 1.055e-34
alpha = 7.297e-3

M0 = 1.988 * 10**30
MeV = 1.60218e-19*1e6

# Nuclear mass
m_N = 939.57*MeV/c**2
# pion decay constant
f_pi = 130.2/sqrt(2)
# pion mass
m_pi = 134.98
m_pipm = 139.57
m_K0 = 497.61
m_Kpm = 493.68
m_eta = 547.86

Dm_EM = sqrt(m_pipm**2 - m_pi**2)
Dm = sqrt(m_K0**2 - (m_Kpm**2 - Dm_EM**2))
m_S = sqrt((3*m_eta**2 - m_pi**2)/2)


e = sqrt(4*pi*alpha)
C = 1/2 * f_pi**2 / e**2 * Dm**2
D = Dm_EM**2 / m_pi**2

f_pi_SI = f_pi*MeV
m_pi_SI = m_pi*MeV/c**2
m_pipm_SI = m_pipm*MeV/c**2



def get_const_fermi_gas():
    """Constants fermi gas"""
    u0 = m_N**4 / (8 * pi**2) * (c**5 / hbar**3) 
    m0 = c**4 / sqrt(4*pi/3 * u0 * G**3) / M0
    r0 = G * m0*M0 / c**2 / 1e3 # (km)
    return u0, m0, r0

def get_const_pion():
    u0 = m_pi_SI**2*f_pi_SI**2 * (c/hbar**3)
    m0 = c**4 / sqrt(4*pi/3 * u0 * G**3) / M0
    r0 = G * m0*M0 / c**2 / 1e3
    return u0, m0, r0


def max_radius_pion_star():
    _, _, r0 = get_const_pion()
    R = pi/sqrt(12) * r0
    print(R)
    print(R / (1 + D))

if __name__=="__main__":
    pass

    # for const in get_const_fermi_gas(): print(const)
    # for const in get_const_pion(): print(const)
    # print("%.3e"%(C/(1e3)**4))
    # u0, _, _ = get_const_pion()
    # print("%.3e"%(C/(m_pi**2*f_pi**2)))
    # print(Δ)

    # max_radius_pion_star()
    # print(sqrt(3/4)*pi)

    # print(( hbar * c / (2* G * m_pi**2) )**(3 / 2) * m_pi / M0)

    print(Dm**4/m_K0**4)