import numpy as np
from numpy import sqrt, pi, log as ln


c = 2.998e8
G = 6.674e-11
hbar = 1.055e-34
alpha = 7.297e-3
e = sqrt(4*pi*alpha)

M0 = 1.988 * 10**30
MeV = 1.60218e-19*1e6

# pion mass
m_pi = 131
m_K0 = 481
m_rho = 770
dm = 0

m_e = 0.5110
m_mu = 105.7

# pion decay constant
f_pi = 128/sqrt(2)
m_S = sqrt(2*m_K0**2 - m_pi**2)


m_pi_SI = m_pi*MeV/c**2
f_pi_SI = f_pi*MeV

# nlo coupling constants
Lr = np.array([
    -3.4,   # H_2
    1.,     # L_1
    1.6,    # L_2
    -3.8,   # L_3
    0.,     # L_4
    1.2,    # L_5
    0.,     # L_6
    0.,     # L_7, not in use. Only here for right indexing
    0.5     # L_8
]) * 1e-3


def get_const_pion():
    u0 = m_pi_SI**2*f_pi_SI**2 * (c/hbar**3)
    m0 = c**4 / sqrt(4*pi/3 * u0 * G**3) / M0
    r0 = G * m0*M0 / c**2 / 1e3
    return u0, m0, r0


def get_const_lepton(m_l):
    u0 = f_pi**2*m_pi**2
    ul0 = m_l**4 / (8*pi**2)
    A = 8 / 3 * ul0/m_l * 1/(u0/m_pi)
    return u0, ul0, A


if __name__=="__main__":
    pass
    print(8*pi**2)