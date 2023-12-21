import numpy as np
from numpy import sqrt

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


# nlo coupling constants
Lr = np.array([
    -3.4,   # H_2
    +1.0,   # L_1
    +1.6,   # L_2
    -3.8,   # L_3
    +0.0,   # L_4
    +1.2,   # L_5
    +0.0,   # L_6
    +0.0,   # L_7, not in use. Only here for right indexing
    +0.5    # L_8
]) * 1e-3

# Lattice constants
m_pi_lattice = 135.
m_K_lattice = 495.
f_pi_lattice = 133. / sqrt(2) 
# f_pi_lattice = 94.


# Renormalization scale
Lambda = 770
