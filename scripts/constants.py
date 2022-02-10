import numpy as np
from numpy import sqrt, pi


c = 2.998e8
G = 6.674e-11
hbar = 1.055e-34
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
     
if __name__=="__main__":

    for const in get_const_fermi_gas(): print(const)
    for const in get_const_pion(): print(const)
