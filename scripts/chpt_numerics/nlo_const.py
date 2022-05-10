import numpy as np
from numpy import pi, sqrt, exp, log as ln
from scipy.optimize import newton
import sys

sys.path.append(sys.path[0] + "/..")


def get_nlo_const(lattice=False):
    if lattice: from constants_lattice import f_pi, m_pi, m_K0, m_rho, m_S, Lr
    else: from constants import f_pi, m_pi, m_K0, m_rho, m_S, Lr

    m_K = m_K0
    M = m_rho


    mpi0_sq = lambda m, mS, f: m**2
    mK0_sq = lambda m, mS, f: 1/2*(m**2 + mS**2)
    meta0_sq = lambda m, mS, f: 1/3*(m**2 + 2*mS**2)


    C = 1/(4*pi)**2
    mpi_nlo_sq = lambda m, mS, f:\
        mpi0_sq(m, mS, f) * (
            1
            + (
                16*Lr[8] - 8*Lr[5] 
                + 24*Lr[6] - 12*Lr[4]
                + C/2 * ln( mpi0_sq(m, mS, f)/M**2 ) 
            ) * mpi0_sq(m, mS, f)/f**2
            + (
                24*Lr[6] - 12*Lr[4] - C/6 * ln( meta0_sq(m, mS, f)/M**2 ) 
            ) * meta0_sq(m, mS, f)/f**2
    )

    mK_nlo_sq = lambda m, mS, f:\
        mK0_sq(m, mS, f) * (
            1
            + 8*(2*Lr[6] - Lr[4]) * mpi0_sq(m, mS, f)/f**2
            + 8*(2*Lr[8] - Lr[5] + 4*Lr[6] - 2*Lr[4]) * mK0_sq(m, mS, f)/f**2 
            + C/3 * ln(meta0_sq(m, mS, f)/M**2) * meta0_sq(m, mS, f)/f**2
    )

    f_nlo_sq = lambda m, mS, f:\
        f**2 *(
            1
            + (8*Lr[4] + 8*Lr[5] - 2*C * ln(mpi0_sq(m, mS, f)/M**2)) \
                * mpi0_sq(m, mS, f)/f**2
            + (16*Lr[4] - C * ln(mK0_sq(m, mS, f)/M**2)) * mK0_sq(m, mS, f)/f**2
    )

    m0 = m_pi
    mS0 = m_S
    f0 = f_pi
    x0 = np.array([m0, mS0, f0])

    eq_mpi = lambda m, mS, f: m_pi**2 - mpi_nlo_sq(m, mS, f)
    eq_mK = lambda m, mS, f: m_K**2 - mK_nlo_sq(m, mS, f)
    eq_f = lambda m, mS, f: f_pi**2 - f_nlo_sq(m, mS, f)

    eq = lambda x: np.array([eq_mpi(*x), eq_mK(*x), eq_f(*x)])

    m, mS, f = newton(eq, x0, tol=1e-12)
    return m , mS, f


if __name__=="__main__":
    print(get_nlo_const())
    print(get_nlo_const(True))
