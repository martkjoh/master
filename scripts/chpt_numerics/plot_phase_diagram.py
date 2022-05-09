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

import sys
sys.path.append(sys.path[0] + "/..")
from constants import Dm, Dm_EM, m_pi, m_K0

D = Dm/m_pi
m = m_K0/m_pi
mpm = sqrt(m**2 - D**2)
DEM = Dm_EM/m_pi
mpiEM = sqrt(1 + DEM**2)
print(DEM**2)


N = 100

# mu_Kpm
u = lambda x, y : y+x/2
# mu_K0
v = lambda x, y : y-x/2
# mu_I
w = lambda x : x


# pi / Kpm
x1 = np.linspace(1, 3, N)
eq1 = lambda x, y : -u(x,y) + (w(x)**2 - 1 + sqrt((w(x)**2 - 1)**2 + 4*w(x)**2*mpm**2 )) / (2*w(x))
sol1 = fsolve(lambda y: eq1(x1, y), np.ones(N))

# pi / K0
x2 = np.linspace(-1, -3, N)
eq2 = lambda x, y : v(x,y) + (w(x)**2 - 1 + sqrt( (w(x)**2 - 1)**2 + 4*w(x)**2*m**2 )) / (2*w(x))
sol2 = fsolve(lambda y: eq2(x2, y), np.ones(N))

# Kpm/K0
y3 = np.linspace(sqrt(m*mpm), 5, N)
eq3 = lambda x, y : -u(x,y) + (v(x,y)**2 - m**2 + sqrt( (v(x,y)**2 + m**2)**2 - 4*v(x,y)**2*D**2 )) / (2*v(x, y))
sol3 = fsolve(lambda x: eq3(x, y3), np.ones(N))

# Kpm
a = mpm - m
x4 = np.linspace(a, 1, N)
eq4 = lambda x, y : mpm-u(x,y)
sol4 = fsolve(lambda y: eq4(x4, y), np.ones(N))

# K0
x5 = np.linspace(-1, a, N)
eq5 = lambda x, y : m-v(x,y)
sol5 = fsolve(lambda y: eq5(x5, y), np.ones(N))

# pi+
y6 = np.linspace(0, fsolve(lambda y: mpm-u(1, y), 1), N)

# pi-
y7 = np.linspace(0, fsolve(lambda y: m-u(1, y), 1), N)



##############
# EM resutls #
##############

uEM = lambda x, y : sqrt((y+x/2)**2 - DEM**2)
wEM = lambda x : sqrt(x**2 - DEM**2)

# pi / Kpm
x1EM = np.linspace(mpiEM, 3, N)
eq1EM = lambda x, y : -uEM(x,y) + (wEM(x)**2 - 1 + sqrt((wEM(x)**2 - 1)**2 + 4*wEM(x)**2*mpm**2 )) / (2*wEM(x))
sol1EM = fsolve(lambda y: eq1EM(x1EM, y), np.ones(N))

# pi / K0
x2EM = np.linspace(-mpiEM, -3, N)
eq2EM = lambda x, y : -v(x,y) + (wEM(x)**2 - 1 + sqrt( (wEM(x)**2 - 1)**2 + 4*wEM(x)**2*m**2 )) / (2*wEM(x))
sol2EM = fsolve(lambda y: eq2EM(x2EM, y), np.ones(N))

# Kpm/K0
y3EM = np.linspace(sqrt(m*sqrt(mpm**2 + DEM**2)), 5, N)
eq3EM = lambda x, y : -uEM(x,y) + (v(x,y)**2 - m**2 + sqrt( (v(x,y)**2 + m**2)**2 - 4*v(x,y)**2*D**2 )) / (2*v(x, y))
sol3EM = fsolve(lambda x: eq3EM(x, y3EM), np.ones(N))

# Kpm
a = sqrt(mpm**2 + DEM**2) - m
x4EM = np.linspace(a, mpiEM, N)
eq4EM = lambda x, y : mpm-uEM(x,y)
sol4EM = fsolve(lambda y: eq4EM(x4EM, y), np.ones(N))

# K0
x5EM = np.linspace(-mpiEM, a, N)
eq5EM = lambda x, y : m-v(x,y)
sol5EM = fsolve(lambda y: eq5EM(x5EM, y), np.ones(N))

# pi+
y6EM = np.linspace(0, fsolve(lambda y: mpm-uEM(mpiEM, y), 1), N)

# pi-
y7EM = np.linspace(0, fsolve(lambda y: m-v(-mpiEM, y), 1), N)



fig, ax = plt.subplots(figsize=(8, 10))


ax.plot(x1, sol1, "k--")
ax.plot(x1, -sol1, "k--")
ax.plot(x2, sol2, "k--")
ax.plot(x2, -sol2, "k--")
ax.plot(sol3, y3, "k--")
ax.plot(sol3, -y3, "k--")
ax.plot(x4, sol4, "k")
ax.plot(x4, -sol4, "k")
ax.plot(x5, sol5, "k")
ax.plot(x5, -sol5, "k")
ax.plot(np.ones(N), y6, "k")
ax.plot(np.ones(N), -y6, "k")
ax.plot(-np.ones(N), y7, "k")
ax.plot(-np.ones(N), -y7, "k")



plt.text(-0.55, 0, "Vacuum")
plt.text(-0.35, -0.5, "phase")

plt.text(1, 4, "$\\langle K^+\\rangle$")
plt.text(-1.9, -4.6, "$\\langle K^-\\rangle$")
plt.text(1, -4.6, "$\\langle \\bar K^0\\rangle$")
plt.text(-1.9, 4, "$\\langle K^0\\rangle$")
plt.text(1.5, 0, "$\\langle \pi^+\\rangle$")
plt.text(-2.4, 0, "$\\langle \pi^-\\rangle$")

ax.set_xlim(-3, 3)
ax.set_ylim(-5, 5)
ax.set_xlabel("$\\mu_I / m_\pi$")
ax.set_ylabel("$\\mu_S / m_\pi$")

fig.savefig("figurer/phase_diagram.pdf", bbox_inches="tight")



def add_plot(ax):
    ax.plot(x1, sol1, "k--")
    ax.plot(x1, -sol1, "k--")
    ax.plot(x2, sol2, "k--")
    ax.plot(x2, -sol2, "k--")
    ax.plot(sol3, y3, "k--")
    ax.plot(sol3, -y3, "k--")
    ax.plot(x4, sol4, "k")
    ax.plot(x4, -sol4, "k")
    ax.plot(x5, sol5, "k")
    ax.plot(x5, -sol5, "k")
    ax.plot(np.ones(N), y6, "k")
    ax.plot(np.ones(N), -y6, "k")
    ax.plot(-np.ones(N), y7, "k")
    ax.plot(-np.ones(N), -y7, "k")

    ax.plot(x1EM, sol1EM, "r--")
    ax.plot(x1EM, -sol1EM, "r--")
    ax.plot(x2EM, sol2EM, "r--")
    ax.plot(x2EM, -sol2EM, "r--")
    ax.plot(sol3EM, y3EM, "r--")
    ax.plot(sol3EM, -y3EM, "r--")
    ax.plot(x4EM, sol4EM, "r")
    ax.plot(x4EM, -sol4EM, "r")
    ax.plot(x5EM, sol5EM, "r")
    ax.plot(x5EM, -sol5EM, "r")
    ax.plot(mpiEM*np.ones(N), y6EM, "r")
    ax.plot(mpiEM*np.ones(N), -y6EM, "r")
    ax.plot(-mpiEM*np.ones(N), y7EM, "r")
    ax.plot(-mpiEM*np.ones(N), -y7EM, "r")


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(2, 2, (1, 3))


ax2 = fig.add_subplot(2, 2, 4)
ax3 = fig.add_subplot(2, 2, 2)

add_plot(ax)
add_plot(ax2)
add_plot(ax3)

ax.set_xlim(-3, 3)
ax.set_ylim(-5, 5)
ax.set_xlabel("$\\mu_I / m_\pi$")
ax.set_ylabel("$\\mu_S / m_\pi$")

ax2.set_xlim(0.8, 1.6)
ax2.set_ylim(3.1, 3.4)

ax3.set_xlim(-0.2, 0.2)
ax3.set_ylim(3.5, 3.8)

plt.tight_layout()
fig.savefig("figurer/phase_diagram_EM.pdf", bbox_inches="tight")
