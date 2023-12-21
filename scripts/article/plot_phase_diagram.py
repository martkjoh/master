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
from constants_phys import f_pi, m_Kpm, m_K0, e, m_pipm
from constants_LO import Dm, Dm_EM, m_pi0, m_K00

D = Dm/m_pipm
m_K00 = m_K00/m_pipm
m_pi0 = m_pi0/m_pipm
m_Kpm0 = m_Kpm0 / m_pipm
DEM = Dm_EM/m_pipm
m_pipm = m_pipm/m_pipm
print(DEM**2)


N = 100

# mu_Kpm
u = lambda x, y : y+x/2
# mu_K0
v = lambda x, y : y-x/2
# mu_I
w = lambda x : x


# pi / Kpm
x1 = np.linspace(m_pi0, 3, N)
eq1 = lambda x, y : -u(x,y) + (w(x)**2 - m_pi0**2 + sqrt((w(x)**2 - m_pi0**2)**2 + 4*w(x)**2*m_Kpm0**2 )) / (2*w(x))
sol1 = fsolve(lambda y: eq1(x1, y), np.ones(N))

# pi / K0
x2 = np.linspace(-m_pi0, -3, N)
eq2 = lambda x, y : +v(x,y) + (w(x)**2 - m_pi0**2 + sqrt((w(x)**2 - m_pi0**2)**2 + 4*w(x)**2*m_K00**2 )) / (2*w(x))
sol2 = fsolve(lambda y: eq2(x2, y), np.ones(N))

# Kpm/K0
y3 = np.linspace(sqrt(m_K00*m_Kpm0), 5, N)
eq3 = lambda x, y : -u(x,y) + (v(x,y)**2 - m_K00**2 + sqrt( (v(x,y)**2 + m_K00**2)**2 - 4*v(x,y)**2*D**2 )) / (2*v(x, y))
sol3 = fsolve(lambda x: eq3(x, y3), np.ones(N))

# Kpm
a = m_Kpm0 - m_K00
x4 = np.linspace(a, m_pi0, N)
eq4 = lambda x, y : m_Kpm0-u(x,y)
sol4 = fsolve(lambda y: eq4(x4, y), np.ones(N))

# K0
x5 = np.linspace(-m_pi0, a, N)
eq5 = lambda x, y : m_K00-v(x,y)
sol5 = fsolve(lambda y: eq5(x5, y), np.ones(N))

# pi+
y6 = np.linspace(0, fsolve(lambda y: m_Kpm0-u(m_pi0, y), m_pi0), N)

# pi-
y7 = np.linspace(0, fsolve(lambda y: m_K00-u(m_pi0, y), m_pi0), N)



##############
# EM resutls #
##############

# mu_Kpm,eff
uEM = lambda x, y : sqrt((y+x/2)**2 - DEM**2)
# mu_I,eff
wEM = lambda x : sqrt(x**2 - DEM**2)

# pi / Kpm
x1EM = np.linspace(m_pipm, 3, N)/m_pipm
eq1EM = lambda x, y : -uEM(x,y) + (wEM(x)**2 - m_pi0**2 + sqrt((wEM(x)**2 - m_pi0**2)**2 + 4*wEM(x)**2*m_Kpm0**2 )) / (2*wEM(x))
sol1EM = fsolve(lambda y: eq1EM(x1EM, y), np.ones(N))

# pi / K0
x2EM = np.linspace(-m_pipm, -3, N)/m_pipm
eq2EM = lambda x, y : -v(x,y) + (wEM(x)**2 - m_pi0**2 + sqrt( (wEM(x)**2 - m_pi0**2)**2 + 4*wEM(x)**2*m_K00**2 )) / (2*wEM(x))
sol2EM = fsolve(lambda y: eq2EM(x2EM, y), np.ones(N))

# Kpm/K0
y3EM = np.linspace(sqrt(m_K00*sqrt(m_Kpm0**2 + DEM**2)), 5, N)
eq3EM = lambda x, y : -uEM(x,y) + (v(x,y)**2 - m_K00**2 + sqrt( (v(x,y)**2 + m_K00**2)**2 - 4*v(x,y)**2*D**2 )) / (2*v(x, y))
sol3EM = fsolve(lambda x: eq3EM(x, y3EM), np.ones(N))

# Kpm
a = sqrt(m_Kpm0**2 + DEM**2) - m_K00
x4EM = np.linspace(a, m_pipm, N)/m_pipm
eq4EM = lambda x, y : m_Kpm0-uEM(x,y)
sol4EM = fsolve(lambda y: eq4EM(x4EM, y), np.ones(N))

# K0
x5EM = np.linspace(-m_pipm, a, N)/m_pipm
eq5EM = lambda x, y : m_K00-v(x,y)
sol5EM = fsolve(lambda y: eq5EM(x5EM, y), np.ones(N))

# pi+
y6EM = np.linspace(0, fsolve(lambda y: m_Kpm0-uEM(m_pipm, y), 1), N)/m_pipm

# pi-
y7EM = np.linspace(0, fsolve(lambda y: m_K00-v(-m_pipm, y), 1), N)/m_pipm



# fig, ax = plt.subplots(figsize=(6, 7))


# ax.plot(x1, sol1, "k--")
# ax.plot(x1, -sol1, "k--")
# ax.plot(x2, sol2, "k--")
# ax.plot(x2, -sol2, "k--")
# ax.plot(sol3, y3, "k--")
# ax.plot(sol3, -y3, "k--")
# ax.plot(x4, sol4, "k")
# ax.plot(x4, -sol4, "k")
# ax.plot(x5, sol5, "k")
# ax.plot(x5, -sol5, "k")
# ax.plot(m_pi0*np.ones(N), y6, "k")
# ax.plot(m_pi0*np.ones(N), -y6, "k")
# ax.plot(-m_pi0*np.ones(N), y7, "k")
# ax.plot(-m_pi0*np.ones(N), -y7, "k")


# ax.text(-0.75, -.50, "Vacuum\n  phase")

# add_text(ax)
# ax.set_xlim(-3, 3)
# ax.set_ylim(-5, 5)
# ax.set_xlabel("$\\mu_I / m_{\pi^0}$")
# ax.set_ylabel("$\\mu_S / m_{\pi^0}$")

# # fig.savefig("figurer/phase_diagram.pdf", bbox_inches="tight")
# plt.show()

def add_text(ax):
    ax.text(0.8, 4, "$\\langle K^+\\rangle$")
    ax.text(-2.1, -4.6, "$\\langle K^-\\rangle$")
    ax.text(0.8, -4.6, "$\\langle \\bar K^0\\rangle$")
    ax.text(-1.9, 4, "$\\langle K^0\\rangle$")
    ax.text(1.5, 0, "$\\langle \pi^+\\rangle$")
    ax.text(-2.4, 0, "$\\langle \pi^-\\rangle$")



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
    ax.plot(m_pi0*np.ones(N), y6, "k")
    ax.plot(m_pi0*np.ones(N), -y6, "k")
    ax.plot(-m_pi0*np.ones(N), y7, "k")
    ax.plot(-m_pi0*np.ones(N), -y7, "k")

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
    ax.plot(m_pipm*np.ones(N), y6EM, "r")
    ax.plot(m_pipm*np.ones(N), -y6EM, "r")
    ax.plot(-m_pipm*np.ones(N), y7EM, "r")
    ax.plot(-m_pipm*np.ones(N), -y7EM, "r")


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(2, 2, (1, 3))

ax.text(-.9 , -.40, "Normal\n phase")
add_text(ax)

ax2 = fig.add_subplot(2, 2, 4)
ax3 = fig.add_subplot(2, 2, 2)

ax2.text(0.75, 2.95, "Normal\n phase")
ax2.text(0.93, 3.14, "$\\langle K^+\\rangle$")
ax2.text(1.1, 2.97, "$\\langle \\pi^+\\rangle$")

ax3.text(-0.12, 3.42, "Normal\n phase")
ax3.text(0.08, 3.6, "$\\langle K^+\\rangle$")
ax3.text(-0.2, 3.6, "$\\langle K^0 \\rangle$")


add_plot(ax)
add_plot(ax2)
add_plot(ax3)

ax.set_xlim(-3, 3)
ax.set_ylim(-5, 5)
ax.set_xlabel("$\\mu_I / m_{\pi^\\pm}$")
ax.set_ylabel("$\\mu_S / m_{\pi^\\pm}$")

ax2.set_xlim(0.7, 1.3)
ax2.set_ylim(2.9, 3.2)

ax3.set_xlim(-0.3, 0.3)
ax3.set_ylim(3.4, 3.7)

# plt.tight_layout()
fig.savefig("article/figurer/phase_diagram_EM_new.pdf", bbox_inches="tight")
# plt.show()
