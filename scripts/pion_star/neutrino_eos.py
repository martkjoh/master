import numpy as np
from numpy import pi, sqrt, arcsinh
from scipy.optimize import newton
from matplotlib import pyplot as plt
import sys

sys.path.append(sys.path[0] + "/..")
from constants import get_const_lepton, f_pi, m_e, m_mu, m_pi
from integrate_tov import get_u

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)


# x = mu_\ell
xf = lambda x: sqrt(x**2 - 1)
pl = lambda x: 1 / 3 * ((2*xf(x)**3 - 3*xf(x)) * x + 3*arcsinh(xf(x))) 
ul = lambda x: (2*xf(x)**3 + xf(x)) * x - arcsinh(xf(x))

# x = mu_I
p_pi = lambda  x: 1/2 * (x - 1/x)**2
u_pi = lambda x: 1/2 * (2 + x**2 - 3/x**2)


def const():
    _, ue0, Ae = get_const_lepton(m_e)
    _, umu0, Amu = get_const_lepton(m_mu)
    a = m_e/m_mu
    u0 = f_pi**2*m_pi**2
    return a, Ae, Amu, ue0, umu0, u0


a, Ae, Amu, ue0, umu0, u0 = const()

def eq(x, y): # x = mu_I/m_\pi, y = mu_e / m_e
    a, Ae, Amu, ue0, umu0, u0 = const()
    more = a*y > 1
    assert np.all(np.diff(more)>=0) # more should only go from 0 to 1
    y2 = np.ones_like(y)/a
    y2[more] = y[more]
    sgn = -1
    return sgn*(x*(1 - 1/x**4) - Ae * (y**2 - 1)**(3/2) - Amu * ((a*y2)**2 - 1)**(3/2))


def mu_e(x):
    _, ue0, Ae = get_const_lepton(m_e)
    return sqrt(1 + 1/Ae**(2/3) * (x * (1-1/x**4))**(2/3))



def plot_mu():
    N = 1000
    x = np.linspace(1, 1.4, N)
    y0 = mu_e(x)

    f = lambda y: eq(x, y)
    y, a, _ = newton(f, y0, full_output=True)
    assert np.all(a)


    plt.plot(x, y)
    plt.plot(x, mu_e(x), "k--")
    plt.show()





