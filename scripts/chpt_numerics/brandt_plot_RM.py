import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse
from numpy import sqrt

sys.path.append(sys.path[0] + "/..")
from integrate_tov import get_u 
from constants import get_const_pion
from pion_star.plot import load_sols, get_data


plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)


def get_MR(data):
    M = data[0]
    err_M = sqrt(data[1]**2 + data[2]**2)
    R = data[3]
    err_R = sqrt(data[4]**2 + data[5]**2)

    return M, R, err_M, err_R


def load_data(name=""):
    sols = np.load("pion_star/data/sols"+name+".npy", allow_pickle=True)
    n = 3
    data = [[] for _ in range(n)]
    u0, m0, r0 = get_const_pion()

    for i, s in enumerate(sols):
        data[0].append(s["R"]*r0)
        data[1].append(s["M"]*m0)
        data[2].append(s["pc"])

    return data


def fill_ellipses(ax, x, y, dx, dy, color,zorder):
    for i in range(len(x)):
        xi, yi = x[i], y[i]
        a, b = 2*dx[i], 2*dy[i]
        e = Ellipse((xi, yi), width=a, height=b,zorder=zorder)
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor(color)

def compare_all():

    fig, ax = plt.subplots(figsize=(16, 8)) 

    names = ["data_brandt/data_p.txt", "data_brandt/data_pe.txt", "data_brandt/data_pm.txt"]
    colors= ["cornflowerblue", "mediumseagreen", "indianred", "plum"]
    N = 4
    alpha = 1.
    labels = ["$\\pi$", "$\\pi e$", "$\\pi \\mu$"]


    for i, name in enumerate(names):
        data = np.loadtxt(name, skiprows=1, unpack=True)
        M, R, err_M, err_R = get_MR(data)
        fill_ellipses(ax, R, M, err_R, err_M, colors[i], zorder=i)
        ax.plot(R, M, color=colors[i], label=labels[i], zorder=i, lw=3)

    name = "data_brandt/MR_pilnu.txt"
    data = np.loadtxt(name, unpack=True)
    M, err_M, R, err_R, stable = data
    label="$\\pi\\ell\\nu_\\ell$"
    fill_ellipses(ax, R, M, err_R, err_M, colors[-1], 4)
    ax.plot(R, M, color=colors[-1], label=label, lw=3)



    names = ["_nlo", "_e", "_mu", "_neutrino_nlo"]
    u0, m0, r0 = get_const_pion()

    for name in names:
        data = load_data(name)
        R = data[0]
        M = data[1]
        ax.plot(R, M, "k--", lw=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$R [\\mathrm{km}]$")
    ax.set_ylabel("$M / M_\odot$")
    ax.set_ylim(10**(-.9), 10**(2.5))
    ax.legend()

    fig.savefig("figurer/pion_star/mass_radius_brandt_all.pdf", bbox_inches="tight")
    # plt.show()


def plot_compare_lattice(t=""):
    fig, ax = plt.subplots(figsize=(18, 10))

    if t=="":
        data = np.loadtxt("data_brandt/data_p.txt", skiprows=1, unpack=True)
        M, R, err_M, err_R = get_MR(data)
    else:
        data = np.loadtxt("data_brandt/MR_pilnu.txt", unpack=True)
        M, err_M, R, err_R, stable = data

    fill_ellipses(ax, R, M, err_R, err_M, "lightblue", 1)
    ax.plot(R, M, color="lightblue", label="Brandt et.al.",lw=5)

    names = [t, t+"_nlo", t+"lattice", t+"_nlo"+"lattice"]
    sols = [load_sols(name) for name in names]
    labels = ["LO, PDG", "NLO, PDG", "LO, lattice", "NLO, lattice"]
    line = ["k-", "r-","k--", "r--"]
    lat = [False, False, True, True]
    markers = ["x", "*", "o", "+"]

    for i, sol in enumerate(sols):
        if lat[i]:from constants_lattice import get_const_pion
        else: from constants import get_const_pion
        u0, m0, r0 = get_const_pion()

        data = get_data(sol)

        R, M, pc = [np.array(d) for d in data]
        x, y, z = R*r0, M*m0, np.log10(pc)
        ax.plot(x, y, line[i], label=labels[i])

        j = np.argmax(M)
        label ="$(M, R) = "+"(%.3f" %(M[j]*m0)+"\, M_\odot, %.3f" %(R[j]*r0)+"\, \mathrm{km})$ "
        ax.plot(R[j]*r0, M[j]*m0, "k", marker=markers[i], ls="", ms=10, label=label)

    ax.set_xlabel("$R [\\mathrm{km}]$")
    ax.set_ylabel("$M / M_\odot$")
    
    plt.legend()
    fig.savefig("figurer/pion_star/lattice_const_compare"+t+".pdf", bbox_inches="tight")

def brandt_neutrino():
    fig, ax = plt.subplots(figsize=(16, 8))
    name = "data_brandt/MR_pilnu.txt"
    data = np.loadtxt(name, unpack=True)
    M, err_M, R, err_R, stable = data
    
    alpha=1.
    label="$\\pi\ell\\nu_\\ell$"

    for i in range(len(R)):
        x, y = (R[i], M[i])
        a, b = (2*err_R[i], 2*err_M[i])
        e = Ellipse((x, y), width=a, height=b)
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(alpha)
        e.set_facecolor(color)

    ax.set_xlim(25, 225)
    ax.set_ylim(0, 30)
    plt.show()


# brandt_neutrino()

compare_all()

# plot_compare_lattice()
# plot_compare_lattice(t="_neutrino")

