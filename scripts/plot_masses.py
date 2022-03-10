import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import sys

from sympy import lambdify
from numpy import arccos, sqrt, cos, sin
from matplotlib import cm

sys.path.append(sys.path[0] + "/..")
from constants import f_pi_MeV

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)



def alpha(mu):
    a = np.zeros_like(mu)
    mask = mu>1
    a[mask] = arccos(1 / mu[mask]**2)
    return a

m1sq = lambda muI, muS : cos(alpha(mu)) - muI**2*cos(alpha(muI))**2
m2sq = lambda muI, muS : cos(alpha(mu)) - muI**2*cos(2*alpha(muI))
m3sq = lambda muI, muS : cos(alpha(mu)) + muI**2*sin(alpha(muI))**2


fig, ax = plt.subplots()

mu = np.linspace(0, 2, 200)

ax.plot(mu, m1sq(mu, 0))
ax.plot(mu, m2sq(mu, 0))
ax.plot(mu, m3sq(mu, 0))


plt.show()