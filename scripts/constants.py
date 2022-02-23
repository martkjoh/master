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
f_pi_MeV = 130.2/sqrt(2)
f_pi = f_pi_MeV*MeV
# pion mass
m_pi_MeV = 134.98
m_pi = m_pi_MeV*MeV/c**2
m_pipm_MeV = 139.57
m_pipm = m_pipm_MeV*MeV/c**2

e = sqrt(4*pi*alpha)

Δm_MeV = sqrt(m_pipm_MeV**2 - m_pi_MeV**2)
C = 1/2 * f_pi_MeV**2 / e**2 * Δm_MeV**2

Δ = Δm_MeV**2 / m_pi_MeV**2


def get_const_fermi_gas():
    """Constants fermi gas"""
    u0 = m_N**4 / (8 * pi**2) * (c**5 / hbar**3) 
    m0 = c**4 / sqrt(4*pi/3 * u0 * G**3) / M0
    r0 = G * m0*M0 / c**2 / 1e3 # (km)
    return u0, m0, r0

def get_const_pion():
    u0 = m_pi**2*f_pi**2 * (c/hbar**3)
    m0 = c**4 / sqrt(4*pi/3 * u0 * G**3) / M0
    r0 = G * m0*M0 / c**2 / 1e3
    return u0, m0, r0


def max_radius_pion_star():
    _, _, r0 = get_const_pion()
    R = pi/sqrt(12) * r0
    print(R)

if __name__=="__main__":

    # for const in get_const_fermi_gas(): print(const)
    # for const in get_const_pion(): print(const)
    # print(C / (m_pi_MeV**2 * f_pi_MeV**2))
    # print(Δ)

    max_radius_pion_star()
    # print(sqrt(3/4)*pi)