import sys
sys.path.append(sys.path[0] + "/..")
from constants_phys import m_pipm, f_pi, f_pi_lattice, m_pi_lattice
import numpy as np
import matplotlib.pyplot as plt

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)



# We are in the isospin, e=0 limit, and choose the charged mass

m_pi = m_pi_lattice
f_pi = f_pi_lattice

def P0(muI):
    return 1 / 2 * f_pi**2 * muI**2 * (1 - m_pi**2 / muI**2) **2 / (f_pi**2*m_pi**2)


def get_therm_LO():
    p0 = m_pi**2 * f_pi**2
    n0 = p0 / m_pi
    
    muI = np.linspace(1., 1.8, 10_000) * m_pi
    x = muI/m_pi 

    PNLO = P0(muI)
    dx = x[1] - x[0]
    nNLO = (PNLO[2:] - PNLO[:-2])  / (2* dx)
    x2 = x[1:-1]

    epsNLO = - PNLO[1:-1] + nNLO * x2
    cs2 = (PNLO[3:-1] - PNLO[1:-3]) / (epsNLO[2:] - epsNLO[:-2])
    x3 = x[2:-2]

    return (x, x2, x2, x3), ( PNLO, nNLO, epsNLO, cs2)

# x, y, sl = get_therm_LO()

# plt.plot(x[1:], y[3])
# plt.show()




# P0()