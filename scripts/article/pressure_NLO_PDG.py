import numpy as np
from numpy import sqrt, pi, log as ln

import sys
sys.path.append(sys.path[0] + "/..")
from constants_NLO_PDG import mpi0, mK0, f0, m_pi, m_K, f_pi, Lambda
from constants_phys import Lr


# Pressure/energy scale
p0 = f_pi**2 * m_pi**2


def P0NLO(muI):
    return 1/2 * f0**2 * muI**2 * (1 - mpi0**2/muI**2)**2 

# Prescription for branch cutting
# ie = np.float128(1) * 
ie = (-1e-15j)

mK0 = m_K
mpi0 = m_pi

mS = sqrt(m_K**2 - 1 / 2 * m_pi**2 ) # \tilde m_{K, 0}
meta = sqrt( 1 / 3 * (4 * m_K**2 - m_pi**2) ) # m_{\eta, 0}

m22 = lambda muI : muI**2 * (1 - m_pi**4 / muI**4) + ie
m32 = lambda muI : muI**2 + 0j
m42 = lambda muI : mS**2 +  1/4 * muI**2 * (1 + m_pi**4 / muI**4  ) + 0j
m82 = lambda muI : meta**2 - 1/3 * m_pi**2 * (1 - m_pi**2 / muI**2) + 0j

def F(z):
    z = z + ie
    return 16/5 * (
        ( (3*z**2 - 10*z - 8) * (1 - sqrt(1 - z)) ) / z**4
        + (z**2 + 4) / z**3
        - 3 * (z**2 - 4 * z + 8) / z**3 
        * ln( (1 + sqrt(1 - z)) / 2 )
    )


def P11(muI):
    return (
        4*Lr[1] + 4*Lr[2] + 2*Lr[3] 
        + 1 / (4 * (4*pi)**2)
        * (
            ln(Lambda**2 / m22(muI))
            +ln(Lambda**2 / m32(muI))
            +1/4 * ln(Lambda**2 / m42(muI))
            + 9/8
        )
    )*muI**4


def P12(muI):
    return -(
        32 * Lr[6] + 1/(4*pi)**2 * (
            ln(Lambda**2 / m_K**2)
            + 2/9 * ln(Lambda**2 / meta**2)
            + 11/18
        )
    ) * m_pi**2*mS**2



def P13(muI):
    return -(
        8*Lr[1] + 8*Lr[2] + 4*Lr[3] - 8*Lr[4] - 4*Lr[5] + 16*Lr[6] + 8*Lr[8]
        + 1/(4 * (4*pi)**2)*(
            3 * ln(Lambda**2 / m_pi**2)
            + ln(Lambda**2 / mS**2)
            - 1/2 * ln(Lambda**2 / m42(muI))
            + 1/9 * ln(Lambda**2 / meta**2)
            + 65/36
        )
    )* m_pi**4

def P14(muI):
    return 1/(4*pi)**2 * (
        ln(m_K**2 / m42(muI))
        + 4/9 * ln(meta**2 / m82(muI))
    ) * mS**4



def P15(muI):
    return -(
        8*Lr[4] - 32*Lr[6] 
        - 1/(2 * (4*pi)**2) *(
            ln(Lambda**2 / m42(muI))
            + 4/9 * ln(Lambda**2 / m82(muI))
            + 13/18
        )
    )* m_pi**4 * mS**2 / muI**2


def P16(muI):
    return (
        4*Lr[1] + 4*Lr[2] + 2*Lr[3] - 8*Lr[4] - 4*Lr[5] + 16*Lr[6] + 8*Lr[8]
        + 1 / (144 * (4*pi)**2) * (
            36 * ln(Lambda**2 / m22(muI))
            + 9 * ln(Lambda**2 / m42(muI))
            + 4 * ln(Lambda**2 / m82(muI))
            - 47 / 2
        )
    ) * m_pi**8 / muI**4


def P17(muI):
    return (
        8*Lr[4] + 1 / (2 * (4*pi)**2) * (ln(Lambda**2 / m42(muI)) + 1/2)
    )*mS**2*muI**2



def P18(muI):
    return -(
        5*m_pi**12 / (12*(4*pi)**2 * (muI**4 - m_pi**4)*muI**4)
        * F( -4 *m_pi**4 / (muI**4 - m_pi**4 ) ) 
    )


def P1_a(muI):
    return (
        P0NLO(muI)
        + P11(muI)
        + P12(muI)
        + P13(muI)
        + P14(muI)
        + P15(muI)
        + P16(muI)
        + P17(muI)
        + P18(muI)
    )


def P1(muI):
    return (P1_a(muI) - P1_a(np.float128(m_pi + 1e-10))) / p0


def get_therm(lim=1.8):
    p0 = m_pi**2 * f_pi**2
    n0 = p0 / m_pi

    muI = np.linspace(1., lim, 10_000, dtype=np.float128) * m_pi + 1e-4
    x = muI/m_pi 

    PNLO = P1(muI).real
    dx = x[1] - x[0]
    nNLO = (PNLO[2:] - PNLO[:-2])  / (2* dx)
    x2 = x[1:-1]

    epsNLO = - PNLO[1:-1] + nNLO * x2
    cs2 = (PNLO[3:-1] - PNLO[1:-3]) / (epsNLO[2:] - epsNLO[:-2])
    x3 = x[2:-2]

    return (x, x2, x2, x3), ( PNLO, nNLO, epsNLO, cs2)


# print(P1(135.1))
# print(P1(200))
# print(P1_a(135.1)/p0)
