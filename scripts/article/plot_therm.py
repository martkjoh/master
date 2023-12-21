import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib import cm
from numpy import sqrt

sys.path.append(sys.path[0] + "/..")
from pressure_LO import get_therm_LO
from pressure_NLO import get_therm
from constants_NLO import m_pi, m_K, f_pi



plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)


def plot1():
    mus = [1.001, 2]
    for mu in mus:
        muI = np.linspace(.998, mu, 1000, dtype=np.float128) * m_pi + 1e-10


        PLO = P0(muI)
        PNLO = P1(muI).real

        fig, ax = plt.subplots(figsize=(15,10))
        ax.plot(muI, PLO, 'k--')
        ax.plot(muI, PNLO, 'r-.')
        ax.set_title("$P$, three flavor")
        plt.savefig("article/figurer/2P_three_flavor" + str(mu) + ".pdf")

        fig, ax = plt.subplots(figsize=(15,10))

        x = muI/m_pi
        dx = x[1] - x[0]
        nLO = np.diff(PLO) / dx
        nNLO = np.diff(PNLO) / dx
        ax.plot(muI[1:], nLO, 'k--')
        ax.plot(muI[1:], nNLO, 'r-.')
        ax.set_title("$n_I$, three flavor")

        plt.savefig("article/figurer/2n_three_flavor" + str(mu)+ ".pdf")


def plot_comp():

    # a = .22 fm
    filenames1 = [
        "data_brandt2/pressure_24x32_MS_0.dat",
        "data_brandt2/nI_24x32_MS_0_int.dat",
        "data_brandt2/energy_density_24x32_MS_0.dat",
        "data_brandt2/cs_24x32_MS_0.dat", 
        ]

    # a = .15 fm
    filenames2 = [
        "data_brandt2/pressure_32x48_MS_0.dat",
        "data_brandt2/nI_32x48_MS_0_int.dat",
        "data_brandt2/energy_density_32x48_MS_0.dat",
        "data_brandt2/cs_32x48_MS_0.dat",
        ]


    fig, ax = plt.subplots(2, 2, figsize=(22, 10))

    ax[0, 0].set_ylabel("$c_s^2$")
    ax[1, 0].set_xlabel("$\\mu_I / m_\pi$")
    ax[1, 1].set_xlabel("$\\mu_I / m_\pi$")
    names = ["p", "nI", "u", "c"]
    label = ["$p", "${n_{I,}}",  "$\\epsilon", "${c_s}"]
    y_label = ["$\mathcal{P}/ \mathcal{P}_0$", "$n_I/n_0$", "$\mathcal{ E}/\mathcal{ E}_0$", "$c_s^2$"]


    res = ['$24^3\\times 32$', "$32^3 \\times 48$"]
    colors = ['blue', 'green']


    xx = np.linspace(1, 1.8)
    ax[1, 1].plot(xx, np.ones_like(xx)/3, ':', color="gray", label="$c_s^2=1/3$") 
    ax[1, 1].legend(loc=2)

    x1, y1 = get_therm()
    x2, y2 = get_therm_LO()
    xs = [x1, x2]
    ys = [y1, y2]
    lss = ['r-.', 'k--']
    label = ["NLO", "LO"]

    for i, name in enumerate(names):

        ax[i%2, i//2].set_ylabel(y_label[i])
        ax[i%2, i//2].set_xlim(.98, 1.8)
        for j in range(2):
            x = xs[j]
            y = ys[j]
            ls = lss[j]
            ax[i%2, i//2].plot(x[i], y[i], ls, label=label[j])


        for j, filenames in enumerate([filenames1, filenames2]):
            eos = np.loadtxt(filenames[i], unpack=True, delimiter=' ')

            u0 = m_pi**2 * f_pi**2
            n0 = m_pi * f_pi**2

            # Here, we change from our units, where cond. happens at mu_I = m_pi and u0 = f_pi^2 m_pi^2,
            # to the lattice units, where cond. happens at mu_I = m_pi / 2 and u0 = m_pi^4.
            units = [[2., m_pi**4 / u0 ], [2., m_pi**3 / n0/ 2], [2., m_pi**4 / u0], [2., 1], ]
            
            un = units[i]
            xb, yb, err = eos[0]*un[0], eos[1]*un[1], eos[2]*un[1]
            ax[i%2, i//2].fill_between(xb, yb+err, yb-err, color=colors[j], label=res[j], alpha=.45)

    ax[0, 0].legend(loc=2)


    fig.savefig("article/figurer/comparison_with_data.pdf", bbox_inches="tight")


def plot_PDG():
    fig, ax = plt.subplots(figsize=(10, 5))

    from constants_NLO import m_pi, m_K, f_pi
    from pressure_NLO import get_therm
    x1, y1 = get_therm(lim=3)
    ax.plot(x1[0], y1[0], '--', color='k', label="Lattice", lw=2)

    from constants_phys import m_pi_lattice, f_pi_lattice

    from constants_NLO_PDG import m_pi, m_K, f_pi
    from pressure_NLO_PDG import get_therm
    a = m_pi / m_pi_lattice
    # a = 1
    b = m_pi**2 * f_pi**2 / (m_pi_lattice**2 * f_pi_lattice**2 )

    x1, y1 = get_therm(lim=3)
    ax.plot(x1[0] * a, y1[0] * b, '-', color="tab:blue", label='PDG', lw=2)
    ax.set_ylabel("$\mathcal{P} / \mathcal{P_0}$")
    ax.set_xlabel("$\mu_I / m_\pi$")
    ax.set_xlim(.95, 3)

    ax.legend()

    # plt.show()

    fig.savefig("article/figurer/PDG.pdf", bbox_inches="tight")

plot_PDG()
