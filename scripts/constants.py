import numpy as np
from numpy import sqrt, pi, log as ln


c = 2.998e8
G = 6.674e-11
hbar = 1.055e-34
alpha = 7.297e-3
e = sqrt(4*pi*alpha)

M0 = 1.988 * 10**30
MeV = 1.60218e-19*1e6

# Nuclear mass
m_N = 939.57*MeV/c**2

# pion mass
m_pi = 134.98
m_pipm = 139.57
m_K0 = 497.61
m_Kpm = 493.68
m_eta = 547.86
m_rho = 770

m_e = 0.5110
m_mu = 105.7

# pion decay constant
f_pi = 130.2/sqrt(2)
f_rho = 154


# C = 1/2 * f_pi**2 / e**2 * Dm_EM**2
C = f_pi**2/(2*e**2) * (m_pipm**2 - m_pi**2)
Curech = 3 / (32*pi**2) * m_rho**2*f_rho**2 * ln(f_rho**2 /(f_rho**2 - f_pi**2))

Dm_EM = sqrt(2 * e**2/f_pi**2 * C)
Dm = sqrt(m_K0**2 - (m_Kpm**2 - Dm_EM**2))
# m_S = sqrt((3*m_eta**2 - m_pi**2)/2)
m_S = sqrt(2*m_K0**2 - m_pi**2)


D = Dm_EM**2 / m_pi**2

f_pi_SI = f_pi*MeV
m_pi_SI = m_pi*MeV/c**2
m_pipm_SI = m_pipm*MeV/c**2
m_e_SI = m_e*MeV/c**2

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


def get_const_fermi_gas():
    """Constants fermi gas"""
    u0 = m_N**4 / (8 * pi**2) * (c**5 / hbar**3) 
    m0 = c**4 / sqrt(4*pi/3 * u0 * G**3) / M0
    r0 = G * m0*M0 / c**2 / 1e3 # (km)
    print(m0*M0) 
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

def get_const_lepton(m_l):
    u0 = f_pi**2*m_pi**2
    ul0 = m_l**4 / (8*pi**2)
    # A = 1/(8*pi**2) * m_l**3/(m_pi*f_pi**2)
    A = 8 / 3 * ul0/m_l * 1/(u0/m_pi)
    return u0, ul0, A

def get_D_mu():
    u0, ul0, A = get_const_lepton(m_mu)
    return 8/3*ul0/u0/A

def get_R_mu():
    D_mu = get_D_mu()
    _, _, r0 = get_const_pion()
    R_mu = pi /(sqrt(12)*(1+D_mu)) * r0
    return R_mu


if __name__=="__main__":
    pass

    # for const in get_const_fermi_gas(): print(const)
    # for const in get_const_pion(): print(const)

    # print("%.3e"%(C/(1e3)**4))
    # print("%.3e"%(C2/(1e3)**4))
    # print(Dm_EM)
    # print(m_pi * sqrt(1 + Dm_EM**2/m_pi**2) - m_pi)
    # print(m_pi * sqrt(1 + Dm_EM**2 / m_pi**2) - m_pi)
    # print(Dm**2/m_pi**2)
    # print(m_Kpm * (1 - sqrt(1 - Dm_EM**2 / m_Kpm**2))  )
    # print((m_K0 - sqrt(m_Kpm**2 - Dm_EM**2))  /  1)

    # _, _, A = get_const_lepton(m_e)
    # print("%.4e" % A)

    # _, _, A = get_const_lepton(m_mu)
    # print("%.4e" % A)

    # print(get_R_mu())

    # print(Dm)

    # max_radius_pion_star()
    # u0 = f_pi**2*m_pi**2
    # pmin = u0* (m_pi/f_pi)**2*(1+m_e/m_pi)**4 / (12*pi**2)

    # a = sqrt(u0/pmin)
    # print(a*55)
    # print(a*90)


    # print(M2/M1)

    # print(( hbar * c / (2* G * m_pi**2) )**(3 / 2) * m_pi / M0)

    # print(m_pi*f_pi/(131*90.5))

    # print(Dm**4/m_K0**4)
    # print(4*pi*f_pi/m_pi)

    # print(m_rho/m_pi)

    #####
    # Electromagnetic constants 
    ####

    # print(Curech/1e3**4)
    # print(C/(f_pi**2*m_pi**2))
    # print(Dm_EM)
    # print((sqrt(1+Dm_EM**2/m_Kpm**2)-1)*m_Kpm/m_pi)

