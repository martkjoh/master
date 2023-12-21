import numpy as np
from numpy import sqrt, pi, log as ln
from scipy.optimize import fsolve
import sys

sys.path.append(sys.path[0] + "/..")
from constants_phys import f_pi_lattice, m_pi_lattice, m_K_lattice, Lr, Lambda


from constants_phys import f_pi, m_pipm, m_Kpm
f_pi, m_pi, m_K = f_pi, m_pipm, m_Kpm



meta0 = lambda mpi0, mK0 : sqrt(  1/3 * ( 4 * mK0**2 - mpi0**2 ) ) 

def correction_pi(mpi0, mK0, f0):
    return (
        - ( 
            8*Lr[4] + 8*Lr[5] - 16*Lr[6] - 16*Lr[8] + 1/(2*(4*pi)**2) * ln(Lambda**2 / mpi0**2)
        ) * mpi0**2 / f0**2
        - (Lr[4] - 2*Lr[6]) * 16 * mK0**2 / f0**2
        + meta0(mpi0, mK0)**2 / (6 * f0**2 * (4 * pi)**2) * ln(Lambda**2 / meta0(mpi0, mK0)**2)
    )
 
def correction_K(mpi0, mK0, f0):
    return(
        - (Lr[4] - 2*Lr[6])*8*mpi0**2 / f0**2 
        - (2*Lr[4] + Lr[5] - 4*Lr[6] - 2*Lr[8]) * 8 * mK0**2 / f0**2
        - meta0(mpi0,mK0)**2 / (3 * (4 * pi * f0)**2) * ln(Lambda**2 / meta0(mpi0, mK0)**2)
    )

def correction_f(mpi0, mK0, f0):
    return (
        (8*Lr[4] + 8*Lr[5] + 2 / (4*pi)**2 * ln(Lambda**2 / mpi0**2)) * (mpi0**2 / f0**2)
        + (16*Lr[4] + 1/(4*pi)**2 * ln(Lambda**2 / mK0**2)) * (mK0**2 / f0**2)
    )

def eq(x):
    mpi0, mK0, f0 = x
    return [
        m_pi**2  - mpi0**2 * (1 + correction_pi(mpi0, mK0, f0)),
        m_K**2 - mK0**2 * (1 + correction_K(mpi0, mK0, f0)),
        f_pi**2 - f0**2 * (1 + correction_f(mpi0, mK0, f0))
    ]


# def correction_pi(a,b,c):
#     return (
#         - ( 
#             8*Lr[4] + 8*Lr[5] - 16*Lr[6] - 16*Lr[8] + 1/(2*(4*pi)**2) * ln(Lambda**2 / m_pi**2)
#         ) * m_pi**2 / f_pi**2
#         - (Lr[4] - 2*Lr[6]) * 16 * m_K**2 / f_pi**2
#         + meta0(m_pi, m_K)**2 / (6 * f_pi**2 * (4 * pi)**2) * ln(Lambda**2 / meta0(m_pi, m_K)**2)
#     )
 
# def correction_K(a, b, c):
#     return(
#         - (Lr[4] - 2*Lr[6])*8*m_pi**2 / f_pi**2 
#         - (2*Lr[4] + Lr[5] - 4*Lr[6] - 2*Lr[8]) * 8 * m_K**2 / f_pi**2
#         - meta0(m_pi,m_K)**2 / (3 * (4 * pi * f_pi)**2) * ln(Lambda**2 / meta0(m_pi, m_K)**2)
#     )

# def correction_f(a, b, c):
#     return (
#         (8*Lr[4] + 8*Lr[5] + 2 / (4*pi)**2 * ln(Lambda**2 / m_pi**2)) * (m_pi**2 / f_pi**2)
#         + (16*Lr[4] + 1/(4*pi)**2 * ln(Lambda**2 / m_K**2)) * (m_K**2 / f_pi**2)
#     )

# def eq(x):
#     mpi0, mK0, f0 = x
#     return [
#         m_pi**2  - mpi0**2 - m_pi**2 * correction_pi(mpi0, mK0, f0),
#         m_K**2 - mK0**2 - m_K**2 * correction_K(mpi0, mK0, f0),
#         f_pi**2 - f0**2 - f_pi**2 *  correction_f(mpi0, mK0, f0)
#     ]


x0 = np.array((m_pi, m_K, f_pi))
sol = fsolve(eq, x0)

mpi0, mK0, f0 = sol
# mpi0 = 135.992
# mK0 = 527.539
# f0 = 78.2728
# mpi0 = 140.95

print(x0)
print(sol)
print(1 - x0/sol)
