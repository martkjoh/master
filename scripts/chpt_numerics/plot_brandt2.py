import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse
from numpy import sqrt

sys.path.append(sys.path[0] + "/..")

from constants_lattice import m_pi, f_pi
ml, fl = m_pi, f_pi
fl1 = 136/sqrt(2)
fl2 = 130/sqrt(2)
fl = [fl1, fl2]


from integrate_tov import get_u
from chpt_numerics.chpt_eos import p, u, nI
from constants import m_pi, f_pi


plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)


fig, ax = plt.subplots(2, 2, figsize=(18, 9))

mu, alpha = np.load("pion_star/data/nlo_mu_alpha.npy")
x = mu/m_pi

# a = .22 fm
filenames1 = [
    "data_brandt2/cs_24x32_MS_0.dat",
    "data_brandt2/pressure_24x32_MS_0.dat",
    "data_brandt2/energy_density_24x32_MS_0.dat",
    "data_brandt2/nI_24x32_MS_0_int.dat",
    ]

# a = .15 fm
filenames2 = [
    "data_brandt2/cs_32x48_MS_0.dat",
    "data_brandt2/pressure_32x48_MS_0.dat",
    "data_brandt2/energy_density_32x48_MS_0.dat",
    "data_brandt2/nI_32x48_MS_0_int.dat",
    ]

N=4
colors = [cm.winter(1-(i+1/2)/(N)) for i in range(N)]

ax[0, 0].set_ylabel("$c_s^2$")
ax[1, 0].set_xlabel("$\\mu_I / m_{\pi^0}$")
ax[1, 1].set_xlabel("$\\mu_I / m_{\pi^0}$")
names = ["c",  "p", "u", "nI"]
label = ["${c_s}", "$p", "$\\epsilon", "${n_{I,}}"]
y_label = ["$c_s^2$", "$p/p_0$", "$\\epsilon/\\epsilon_0$", "$n_I/n_0$"]
j = np.where(x>=1)[0][0]

c2 = lambda x : p
lo_func = [p, u, nI]

lo = [0]
for f in lo_func:
    lo.append(np.concatenate( [ np.zeros_like(x[:j]), f(1/x[j:])] ))


res = ['$24^3\\times 32$', "$32^3 \\times 48$"]
colors = ['green', 'blue']


n=915
for i, name in enumerate(names):

    if i!=0:
        y = np.load("pion_star/data/nlo_"+name+".npy")

        ax[i%2, i//2].plot(x, lo[i], "k--", label="$\\mathrm{LO}$")
        ax[i%2, i//2].plot(x, y, "r-.", label="$\\mathrm{NLO}$")
        ax[i%2, i//2].set_ylim(-0.1, 1.1*y[n])
        ax[i%2, i//2].set_xlim(0.9, x[n])
        ax[i%2, i//2].set_ylabel(y_label[i])


    else: 
        p0 = lo[1]
        u0 = lo[2]

        x0 = (x[1:] + x[:-1]) / 2
        y = (p0[1:] - p0[:-1]) / (u0[1:] - u0[:-1])
        ax[i%2, i//2].plot(x0, y, "k--")

        p0 = np.load("pion_star/data/nlo_"+names[1]+".npy")
        u0 = np.load("pion_star/data/nlo_"+names[2]+".npy")
        x0 = (x[1:] + x[:-1]) / 2
        y = (p0[1:] - p0[:-1]) / (u0[1:] - u0[:-1])
        d = 670
        ax[i%2, i//2].plot(x0[d:], y[d:], "r-.")

        ax[i%2, i//2].set_xlim(0.9, x[n])
        ax[i%2, i//2].set_ylim(-.1, .8)
        ax[i%2, i//2].set_ylabel(y_label[i])

        xx = np.linspace(0, 3)
        ax[0, 0].plot(xx, np.ones_like(xx)/3, 'y:', label="$c_s^2=1/3$")

    for j, filenames in enumerate([filenames1, filenames2]):
        eos = np.loadtxt(filenames[i], unpack=True, delimiter=' ')

        # u0 = m_pi**2 * f_pi**2
        # n0 = m_pi * f_pi**2 * 2
        # a = ml / m_pi

        u0 = ml**2 * fl[j]**2
        n0 = ml * fl[j]**2 * 2
        a = 1.

        units = [[2*a, 1], [2*a, ml**4 / u0 ], [2*a, ml**4 / u0], [2*a, ml**3 / n0]]
        
        un = units[i]
        xb, yb, err = eos[0]*un[0], eos[1]*un[1], eos[2]*un[1]
        if i == 2:
            ax[i%2, i//2].fill_between(xb, yb+err, yb-err, color=colors[j], label=res[j], alpha=.6)
        else:
            ax[i%2, i//2].fill_between(xb, yb+err, yb-err, color=colors[j], alpha=.6)

ax[0, 0].legend(loc=2)
ax[0, 1].legend(loc=2)


fig.savefig("figurer/comparison_with_data.pdf", bbox_inches="tight")
# plt.show()
