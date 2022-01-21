import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams['mathtext.fontset'] = 'cm'
font = {'family' : 'serif', 
        'size': 20}
plt.rc('font', **font)
plt.rc('lines', lw=2)


def p(r, R):
    return - (sqrt(1 - 2 * R**2) - sqrt(1 - 2 * r**2 ) ) \
    / (3 * sqrt(1 - 2 *  R**2) - sqrt(1 - 2 * r**2))


R02 = 0.4
R12 = 0.44
N = 20
Rs = np.linspace(sqrt(R02), sqrt(R12), N)


fig, ax = plt.subplots(figsize=(10, 6))

ax.set_xlabel("$r / r_0$")
ax.set_ylabel("$p / u_0$")

ax.set_title("$R^2 \in [" + str( R02 ) + "," + str(R12) + "]$")

for i, R in enumerate(Rs):
    r = np.linspace(0, R, 100)
    plt.plot(r, p(r, R), color = cm.viridis(i / N))

plt.savefig("figurer/incompressible.pdf")
plt.show()
# print(p(0.9, 0.2))
