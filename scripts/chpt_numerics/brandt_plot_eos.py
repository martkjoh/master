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

def get_eos(data):
    p = data[0]*a
    err_p = sqrt(data[1]**2 + data[2]**2)*a
    u = data[3]*a
    err_u = sqrt(data[4]**2 + data[5]**2)*a

    return p, u, err_p, err_u


fig, ax = plt.subplots(figsize=(14, 6))

names = ["data_brandt/EOS_p.txt", "data_brandt/EOS_pe.txt", "data_brandt/EOS_pm.txt"]
colors= ["cornflowerblue", "mediumseagreen", "indianred", "gold"]
N = 4
# colors = [cm.winter(1-(i+1/2)/(N)) for i in range(N)]
alpha=.4
labels = ["$\\pi$", "$\\pi e$", "$\\pi \\mu$"]



def fill_ellipses(ax, x, y, dx, dy, color, zorder):
    for i in range(len(x)):
        xi, yi = x[i], y[i]
        a, b = 2*dx[i], 2*dy[i]
        e = Ellipse((xi, yi), width=a, height=b, zorder=zorder)
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor(color)


for i, name in enumerate(names):
    eos = np.loadtxt(name, skiprows=1, unpack=True)
    p, u, err_p, err_u = get_eos(eos)

    fill_ellipses(ax, p, u, err_p, err_u, colors[i], i)
    ax.plot(p, u, color=colors[i], label=labels[i], zorder=i, lw=4)

names=["", "_e", "_mu"]
p_max = 1
u_max = 2.5
# p_max = 0.0004
# u_max = 0.004

for name in names:
    u_path = "pion_star/data/eos"+name+".npy"
    u = get_u(u_path)
    p = np.linspace(0, p_max, 1000)
    u = np.array([u(p0) for p0 in p])
    ax.plot(p, u, "k--", lw=3)


ax.set_xlim(0, p_max)
ax.set_ylim(0, u_max)
ax.set_xlabel("$p/u_0$")
ax.set_ylabel("$u/u_0$")
ax.legend()

fig.savefig("figurer/brandt_eos.pdf", bbox_inches="tight")
# plt.show()
