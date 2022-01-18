import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams['mathtext.fontset'] = 'cm'
font = {'family' : 'serif', 
        'size': 20}
plt.rc('font', **font)
plt.rc('lines', lw=2)


def p(r, M):
    return - (sqrt(1 - 2*M) - sqrt(1 - 2 * M * r**2 ) ) \
    / (3 * sqrt(1 - 2 * M) - sqrt(1 - 2 * M * r**2))

r = np.linspace(0, 1, 100)

M0 = 0.4
M1 = 0.44
N = 20
Ms = np.linspace(M0, M1, N)


fig, ax = plt.subplots(figsize=(10, 6))

ax.set_xlabel("$r/R$")
ax.set_ylabel("$p / \\rho_0$")

ax.set_title("$M = \\frac{4 \pi}{3} R^3 \\rho_0 G \in [" + str(M0)  + "," + str(M1) + "]$") 

for i, M in enumerate(Ms):
    plt.plot(r, p(r, M), color = cm.viridis(i / N))

plt.savefig("figurer/incompressible.pdf")

