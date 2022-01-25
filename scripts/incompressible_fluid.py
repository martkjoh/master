import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams['mathtext.fontset'] = 'cm'
font = {'family' : 'serif', 
        'size': 20}
plt.rc('font', **font)
plt.rc('lines', lw=2)


def p(r, k):
    return - (sqrt(1 - 2 * k) - sqrt(1 - 2 * k * r**2 ) ) \
    / (3 * sqrt(1 - 2 *  k) - sqrt(1 - 2 * k * r**2))


k0 = 0.4
k1 = 0.44
N = 20
k = np.linspace(k0, k1, N)


fig, ax = plt.subplots(figsize=(10, 6))

ax.set_xlabel("$r / r_0$")
ax.set_ylabel("$p / u_0$")

ax.set_title("$k_1 \in [" + str( k0 ) + "," + str(k1) + "]$")

for i in range(N):
    r = np.linspace(0, 1, 100)
    plt.plot(r, p(r, k[i]), color = cm.viridis(i / N))

plt.savefig("figurer/incompressible.pdf")
plt.show()
