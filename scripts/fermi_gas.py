import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.integrate import quad
from numpy import pi, sqrt,  exp, log as ln


# Particle mass
m = 1
# Chemical potential
mu = lambda x : sqrt(x**2 + 1)

def u(x):
    return m**4 / (8 * pi**2) * (
        mu(x) * x * (2*x**2 + 1) - ln(mu(x) + x)
    )

def p(x):
    return m**4 / (3 * 8 * pi**2) * (
        mu(x) * x * (2*x**2 - 3) + 3*ln(mu(x) + x)
    )


 
def f(p, u):
    return u - 1/3 * p

def dmdr(u, r):
    return  4 * pi * r**2 * u

def dpdr(p, u, m, r):
    return - (4 * pi * r**3 * p - m) * (p + m) / r / (r - 2 * m)


def f(p0, m0, r0, dr=0.001):
    # solve f(p0, u0) for u0
    u0 = 1 / 3*p0
    dmdr0 = dmdr(u0, r0)
    dpdr0 = dpdr(p0, u0, m0, r0)
    r1 = r0 + dr
    m1 = m0 + dmdr0 * dr
    p1 = p0 + dpdr0 * dr
    return p1, m1, r1

N = 10
p0 = 10**np.linspace(-0.5, 3.5, N)
M = np.zeros(N)
R = np.zeros(N)

# print(dpdr(1, 3, 0, 0.01))

for i, p in enumerate(p0):
    m, r = 0, 0.000001,
    while p>=0:
        p, m, r = f(p, m, r)
    M[i] = m
    R[i] = r
    print(i)

print(M)
print(p0)
plt.plot(R, M, "--.")
plt.show()
