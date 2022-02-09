
import numpy as np
import matplotlib.pyplot as plt

from numpy import pi, sqrt, log10 as log
from matplotlib import cm, colors, collections


plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)



def load_sols(name="neutron"):
    return np.load("pion_star/data/sols_"+name+".npy", allow_pickle=True)


def plot_mass_radius(name="tree"):
    sols = load_sols(name)
    N = len(sols)    

    data = [[], [], []]
    for i, s in enumerate(sols):
        data[0].append(s.t[-1])
        data[1].append(s.y[1][-1])
        data[2].append(s.y[0][0])
    R, M, p0 = [np.array(d) for d in data]
    print(R)
    print(M)

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(R, M, "--xk")

    fig.savefig("figurer/mass_radius_pion_star_" + name + ".pdf", bbox_inches="tight")
    
 

def plot_eos():
    pnr = np.linspace(0, .1, 1000)
    pur = np.linspace(0, 50, 1000)
    ps = [pnr, pur]
    fig, ax = plt.subplots(1, 2,figsize=(18, 6))
    for i, p in enumerate(ps):
        u = [u_fermi(p0) for p0 in p]
        ax[i].plot(p, u, label="$ \\tilde u(\\tilde p)$")
        ax[i].plot(p, u_fermi_nonrel(p), "k--", label="$ \\tilde u_{\mathrm{nr}} (\\tilde p)$")
        ax[i].plot(p, 3*p, "r-.", label="$ \\tilde u_{\mathrm{ur}} (\\tilde p)$")
        ax[i].set_xlabel("$p / p_0$")
        ax[i].set_ylabel("$u / u_0$")

    ax[0].legend(loc="upper left")

    fig.savefig("figurer/fermi_eos.pdf", bbox_inches="tight")
    



# plot_norm_pressure_mass()
plot_mass_radius()

# plot_eos()

