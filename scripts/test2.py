import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numpy import sqrt

from matplotlib import cm

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)

from constants import Dm, m_pi, m_K0

D = Dm/m_pi
m = m_K0/m_pi


fig, ax = plt.subplots(figsize=(8, 12))
y = np.linspace(sqrt(m*sqrt(m**2 - D**2)), 5, 100)

u = lambda x, y : y+x/2
v = lambda x, y : y-x/2

eq = lambda x, y : -u(x,y) + (v(x,y)**2 - m**2 + sqrt( (v(x,y)**2 + m**2)**2 - 4*v(x,y)**2*D**2 )) / (2*v(x, y))
f = lambda x: eq(x, y)
sol = fsolve(f, np.ones_like(y))
ax.plot(sol, y, "k--")
ax.plot(sol, -y, "k--")


eq = lambda x, y : -u(x,y) + (x**2 - 1 + sqrt( (x**2 - 1)**2 + 4*x**2*(m**2 - D**2) )) / (2*x)
x = np.linspace(1, 3)
f = lambda y: eq(x, y)
sol = fsolve(f, np.ones_like(x))
ax.plot(x, sol, "k--")
ax.plot(x, -sol, "k--")

eq = lambda x, y : v(x,y) + (x**2 - 1 + sqrt( (x**2 - 1)**2 + 4*x**2*(m**2) )) / (2*x)
x = np.linspace(-1, -3)
f = lambda y: eq(x, y)
sol = fsolve(f, np.ones_like(x))
ax.plot(x, sol, "k--")
ax.plot(x, -sol, "k--")



a = sqrt(m**2 - D**2) - m
x = np.linspace(a, 1, 100)
eq = lambda x, y : sqrt(m**2-D**2)-u(x,y)
f = lambda y: eq(x, y)
sol = fsolve(f, np.ones_like(x))
ax.plot(x, sol, "k")
ax.plot(x, -sol, "k")

x = np.linspace(-1, a, 100)
eq = lambda x, y : m-v(x,y)
f = lambda y: eq(x, y) 
sol = fsolve(f, np.ones_like(x))
ax.plot(x, sol, "k")
ax.plot(x, -sol, "k")

y = np.linspace(0, sqrt(m**2 - D**2) - 1/2)
ax.plot(np.ones_like(y), y, "k")
ax.plot(np.ones_like(y), -y, "k")

y = np.linspace(0, m - 1/2)
ax.plot(-np.ones_like(y), y, "k")
ax.plot(-np.ones_like(y), -y, "k")


plt.text(-0.94, 0, "Normal phase")
plt.text(1, 4, "$\\langle K^+\\rangle$")
plt.text(-1.6, -4.6, "$\\langle K^-\\rangle$")
plt.text(1, -4.6, "$\\langle \\bar K^0\\rangle$")
plt.text(-1.6, 4, "$\\langle K^0\\rangle$")
plt.text(1.2, 0, "$\\langle \pi^+\\rangle$")
plt.text(-1.9, 0, "$\\langle \pi^-\\rangle$")

ax.set_xlim(-3, 3)
ax.set_ylim(-5, 5)
ax.set_xlabel("$\\mu_I / m_\pi$")
ax.set_ylabel("$\\mu_S / m_\pi$")


fig.savefig("figurer/phase_diagram.pdf", bbox_inches="tight")
