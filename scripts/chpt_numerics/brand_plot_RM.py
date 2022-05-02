import numpy as np
import sys
from matplotlib import pyplot as plt
from numpy import sqrt

sys.path.append(sys.path[0] + "/..")
from integrate_tov import get_u 
from constants import get_const_pion


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


fig, ax = plt.subplots(figsize=(16, 10)) 

names = ["data_brandt/data_p.txt", "data_brandt/data_pe.txt", "data_brandt/data_pm.txt"]
colors = ["skyblue", "lightsalmon", "mediumseagreen"]
colors = ["blue", "green", "orange"]
labels = ["$\\pi$", "$\\pi + e$", "$\\pi + \\mu$"]


for i, name in enumerate(names):
    data = np.loadtxt(name, skiprows=1, unpack=True)
    M, R, err_M, err_R = get_MR(data)

    ax.fill_between(R, M-err_M, M+err_M, color=colors[i], label=labels[i], alpha=0.3)
    ax.fill_betweenx(M, R-err_R, R+err_R, color=colors[i], alpha=0.3)

name = "data_brandt/MR_pilnu.txt"
data = np.loadtxt(name, unpack=True)
M, err_M, R, err_R, stable = data
color="purple"
label="$\\pi+\ell+\\nu_\\ell$"
ax.fill_between(R, M-err_M, M+err_M, color=color, label=label, alpha=0.3)
ax.fill_betweenx(M, R-err_R, R+err_R, color=color, alpha=0.3)



names = ["", "_e", "_mu", "_neutrino"]
u0, m0, r0 = get_const_pion()

for name in names:
    data = load_data(name)
    R = data[0]
    M = data[1]
    ax.plot(R, M, "k--")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$R [\\mathrm{km}]$")
ax.set_ylabel("$M / M_\odot$")
ax.legend()

fig.savefig("figurer/pion_star/mass_radius_brandt_all.pdf", bbox_inches="tight")
# plt.show() 
