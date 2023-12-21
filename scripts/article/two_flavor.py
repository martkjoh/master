import numpy as np
from numpy import pi, sqrt, cos, sin, arccos, log as ln
from scipy.optimize import fsolve


m = 135.
f = 133/sqrt(2)
l1 = -0.4
l2 = 4.3
l3 = 2.9
l4 = 4.4

ie = -1e-15j

def eq1(x):
    m0, f0 = x
    return (
        m**2 - m0**2 * (1 - m0**2 / (2 * (4*pi*f0)**2) * l3),
        f**2 - f0**2 * (1 + 2*m0**2 / (4*pi*f0)**2 * l4)
    )

def eq2(x):
    m0, f0 = x
    return (
        m**2 - m0**2 * (1 - m**2 / (2 * (4*pi*f)**2) * l3),
        f**2 - f0**2 * (1 + 2*m**2 / (4*pi*f)**2 * l4)
    )

def eq3(x):
    m0, f0 = x
    return (
        m**2 - (m0**2 - m**2 * m**2 / (2 * (4*pi*f)**2) * l3),
        f**2 - (f0**2 + f**2 * 2*m**2 / (4*pi*f)**2 * l4)
    )


def F(z):
    z = z + ie
    return 16/5 * (
        ( (3*z**2 - 10*z - 8) * (1 - sqrt(1 - z)) ) / z**4
        + (z**2 + 4) / z**3
        - 3 * (z**2 - 4 * z + 8) / z**3 
        * ln( (1 + sqrt(1 - z)) / 2 )
    )

P0 = lambda mu : 1/2 * f**2 * mu**2 * (1 - m**2 / mu**2)**2 / (m**2*f**2)
P1_a = lambda mu : (
    1/2 * f0**2 * mu**2 * (1 - m0**2 / mu**2)**2
    + m**8 / mu**4 / (6 * (4*pi)**2) \
        * (l1 + 2*l2 - 3/2 * l3 - 5/4 + 3/2*ln( m**2 * mu**2 / (mu**4 - m**4) + ie ) )
    + mu**4 / (6 * (4*pi)**2) \
        * ( l1 + 2*l2 + 3/2 + 3/2*ln( m**4 / (mu**4 - m**4) + ie ) )
    - 5 * m**12 / (12*(4*pi)**2 * (mu**4 - m**4) * mu**4) * F( -4 * m**4 / (mu**4 - m**4))
    - m**4 / ( 3 * (4*pi)**2) * (l1 + 2*l2 - 3/4*l3 + 9/8)
) / (m**2*f**2)

P1 = lambda mu : P1_a(mu) - P1_a(m*(1 + 1e-10))




from matplotlib import pyplot as plt
from matplotlib import cm

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)


eqs = [eq1, eq2, eq3]
for eq in eqs:
    x0 = np.array([m, f])
    sol = fsolve(eq, x0)
    m0, f0 = sol
    print(sol)


    mus = [1.001,]
    for mu in mus:
        muI = np.linspace(.999, mu, 1000, dtype=np.float128) * m + 1e-10


        PLO = P0(muI)
        PNLO = P1(muI).real

        fig, ax = plt.subplots(figsize=(15,10))
        ax.plot(muI, PLO, 'k--')
        ax.plot(muI, PNLO, 'r-.')
        ax.set_title("$P$, two flavor")
        # plt.savefig("article/figurer/P_two_flavor" + str(mu) + ".pdf")

        fig, ax = plt.subplots(figsize=(15,10))

        x = muI/m
        dx = x[1] - x[0]
        nLO = np.diff(PLO) / dx
        nNLO = np.diff(PNLO) / dx
        ax.plot(muI[1:], nLO, 'k--')
        ax.plot(muI[1:], nNLO, 'r-.')
        ax.set_title("$n_I$, two flavor")

        # plt.savefig("article/figurer/n_two_flavor" + str(mu)+ ".pdf")

        plt.show()
        