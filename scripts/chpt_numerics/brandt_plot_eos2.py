import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse
from numpy import sqrt

sys.path.append(sys.path[0] + "/..")
from constants_lattice import m_pi, f_pi
from integrate_tov import get_u

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)


a = (f_pi**2/m_pi**2)**(-1)
b=1

def get_eos(data):
    p = data[0]*a
    err_p = sqrt(data[1]**2 + data[2]**2)*a
    u = data[3]*a
    err_u = sqrt(data[4]**2 + data[5]**2)*a

    return p, u, err_p, err_u

names = ["data_brandt/EOS_p.txt", "data_brandt/EOS_pe.txt", "data_brandt/EOS_pm.txt", "data_brandt/EoS_piellnu.txt"]
N = len(names)
colors = [cm.cool(1-(i+1/2)/(N)) for i in range(N)]
alpha=.4
labels = ["$\\pi$", "$\\pi e$", "$\\pi \\mu$", "$\\pi \\ell \\nu$"]
names2=["", "_e", "_mu", "_neutrino"]
nlo = [1, 0, 0, 1]

def fill_ellipses(ax, x, y, dx, dy, color, zorder):
    for i in range(len(x)):
        if i%1!=0: continue
        xi, yi = x[i], y[i]
        a, b = 2*dx[i], 2*dy[i]
        e = Ellipse((xi, yi), width=a, height=b, zorder=zorder)
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor(color)


color = "blue"

for i, name in enumerate(names):
    # if i<3: continue
    eos = np.loadtxt(name, skiprows=1, unpack=True)
    if i == 3:
        p, u, err_p, err_u = eos[1]*a, eos[3]*a, eos[2]*a, eos[4]*a
        mask = p!=0
        p, u, err_p, err_u = p[mask], u[mask], err_p[mask], err_u[mask]
    else:
        p, u, err_p, err_u = get_eos(eos)

    fig, ax = plt.subplots(figsize=(7, 4))

    color = "lightblue"
    ax.plot(p, u, zorder=0, color=color, label=labels[i], lw=2.5)
    fill_ellipses(ax, p, u, err_p, err_u, color, 1)

    u_path = "pion_star/data/eos"+names2[i]+".npy"
    u = get_u(u_path)
    p = np.linspace(0, np.max(p), 1000)

    lw = 2
    ax.plot(p, u(p), "k--", lw=lw, label="LO")
    if nlo[i]:
        u = get_u(u_path.split(".")[0]+"_nlo.npy")
        p = np.linspace(0, np.max(p), 1000)
        ax.plot(p, u(p), "k-.", lw=lw, label="NLO")
    if i==3:
        ax.plot(p, 3*p, ":", color="red", lw=lw, label="$u=3p$")
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 15)

    ax.set_xlabel("$p/u_0$")
    ax.set_ylabel("$u/u_0$")
    ax.legend()

    # plt.show()
    fig.savefig("figurer/brandt_eos"+names2[i]+".pdf", bbox_inches="tight")

