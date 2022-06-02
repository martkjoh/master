import numpy as np
import matplotlib.pyplot as plt
import sys

from numpy import pi, sqrt, log10 as log
from matplotlib import cm, colors, collections

from fermi_gas_eos import u_fermi_nonrel
sys.path.append(sys.path[0] + "/..")
from constants import get_const_fermi_gas 
from integrate_tov import get_u


u0, m0, r0 = get_const_fermi_gas()


plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)



def load_sols(name="neutron"):
    return np.load("fermi_gas_star/data/sols_"+name+".npy", allow_pickle=True)

def get_data(sols):
    data = [[], [], []]
    for s in sols:
        data[0].append(s["R"])
        data[1].append(s["M"])
        data[2].append(s["pc"])
    return data

 
def plot_norm_pressure_mass():
    all_sols = load_sols()
    sols = all_sols[40:160:8]
    N = len(sols)
    
    fig, ax = plt.subplots(2, figsize=(14, 10), sharex=True)
    [a.grid(linestyle="--", alpha=1, lw=0.4) for a in ax]
    pcs = []
    for i, s in enumerate(sols):
        f = s["f"]
        R, M, pc = s["R"], s["M"], s["pc"]
        pcs.append(pc)

        c = cm.viridis(i/N)
        r = np.linspace(0, R, 1000)
        p, m = f(r)
        ax[0].plot(r/R, p/pc, color=c, alpha=0.6)
        ax[1].plot(r/R, m/M, lw=2,  color=c, alpha=0.6)

    M = [s["M"] for s in all_sols]
    pc = [s["pc"] for s in all_sols]
    i = np.argmax(M)
    m = lambda r: all_sols[i]["f"](r)[1]
    p = lambda r: all_sols[i]["f"](r)[0] / pc[i]
    r = np.linspace(0, all_sols[i]["R"], 100)
    ax[0].plot(r/r[-1], p(r)/p(0), "k--", lw=2)
    ax[1].plot(r/r[-1], m(r)/m(r)[-1], "k--", lw=2, label="$M_\\mathrm{max}$")


    ax[0].set_ylabel("$p/p_c$")
    ax[1].set_ylabel("$m/M$")
    ax[1].set_xlabel("$r / R$")

    c = np.arange(1, 5 + 1)

    norm = colors.Normalize(vmin=log(pcs[0]), vmax=log(pcs[-1]))
    cmap = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    cmap.set_array([])

    cb = fig.colorbar(cmap, ax=ax, location="right")
    cb.set_label( label="$\log_{10} [p_c / p_0] $", labelpad=25, rotation=270)

    fig.legend(bbox_to_anchor=(0.73, 0.87))
    fig.savefig("figurer/pressure_mass.pdf", bbox_inches="tight")


def plot_mass_surface():
    sols = load_sols()
    sols = sols[::]
    N = len(sols)
    m = []
    r = np.linspace(0, 1, 100)
    R = []
    for i, s in enumerate(sols):
        R.append(s["R"])
        f = s["f"]
        m.append(f(R[i] * r)[1])  


    m = np.array(m)
    r = np.array(r)
    R = np.array(R)
    r, R = np.meshgrid(r, R)    

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        r, R, m,
        cmap=cm.viridis
        )
    fig.colorbar(surf)
    
    # plt.show()


def plot_mass_radius(name="neutron"):
    sols = load_sols()
    N = len(sols)
    data = get_data(sols)

    R, M, p0 = [np.array(d) for d in data]

    fig, ax = plt.subplots(figsize=(16, 8))

    # Mass-radius limit
    Rs = np.linspace(0, 2.5, 100)
    ax.plot(Rs, 4 / 9 * Rs, "k--", label="$M = \\frac{4}{9} R$")
    
    # plot line (x, y) with z giving values to colormap
    x, y = R*r0, M*m0
    z = log(p0)

    norm = colors.Normalize(z.min(), z.max())
    points = np.array([x, y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
    lc = collections.LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(z)
    line = ax.add_collection(lc)
    cb = fig.colorbar(line, ax=ax)
    cb.set_label( label="$\log_{10} [p_c / p_0] $", labelpad=25, rotation=270)


    i = np.argmax(M)
    label ="$(M, R) = " \
        + "(%.3f" %(M[i]*m0)\
        + "\, M_\odot, %.3f" %(R[i]*r0) \
        + "\, \mathrm{km})$ "
    ax.plot(R[i]*r0, M[i]*m0, "kx", ms=10, label=label)

    R_oppenheimer = [21.1, 13.3, 9.5, 6.8, 3.1]
    M_oppenheimer = [0.4, 0.6, 0.71, 0.64, 0.34]
    ax.plot(R_oppenheimer, M_oppenheimer, "o", ms=10, fillstyle="none", label="Oppenheimer-Volkoff")

    ax.set_xlabel("$R [\\mathrm{km}]$")
    ax.set_ylabel("$M / M_\odot$")
    ax.set_ylim(0, 0.8)
    ax.grid(linestyle="--", alpha=1, lw=0.4)
    fig.legend(bbox_to_anchor=(0.74, 0.88))

    fig.savefig("figurer/mass_radius_" + name + ".pdf", bbox_inches="tight")
    
    
def plot_mass_radius_compare():
    sols1 = load_sols("neutron")
    sols2 = load_sols("neutron_non_rel")
    sols3 = load_sols("neutron_newt")
    sols4 = load_sols("neutron_newt_non_rel")
    sols = [sols1, sols2, sols3, sols4]

    N = len(sols1)
    assert N == len(sols2); assert N ==len(sols3)
    datas = [get_data(s) for s in sols]
        
    fig, ax = plt.subplots(figsize=(16, 8))

    linestyles = ["-", "-.", "--", (0, (6, 1, 1, 1, 1, 1))]
    labels = [
        "Full EOS + TOV",
        "Non-rel. EOS + TOV",
        "Full EOS + Newt. gravity",
        "Non-rel. EOS + Newt. gravity"
        ]

    for i, data in enumerate(datas):
        R, M, p0 = [np.array(d) for d in data]

        # plot line (x, y) with z giving values to colormap
        x, y = R*r0, M*m0
        z = log(p0)

        # hack to get multi-colored line
        # https://nbviewer.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
        norm = colors.Normalize(z.min(), z.max())
        m = 10
        n = len(x) // m
        assert m*n + 1 == len(x) # Are all points included?
        segments = [[[x[j], y[j]] for j in range(i*m, (i+1)*m+1)] for i in range(n)]
        lc = collections.LineCollection(segments, cmap='viridis', norm=norm, ls=linestyles[i], label=labels[i], lw=1.5)
        lc.set_array(z[::m])
        line = ax.add_collection(lc)
        

    cb = fig.colorbar(line)
    cb.set_label( label="$\log_{10} [p_c / p_0] $", labelpad=25, rotation=270)

    ax.set_xlabel("$R [\\mathrm{km}]$")
    ax.set_ylabel("$M / M_\odot$")
    ax.set_ylim(0, 1.8)
    ax.set_xlim(0, 45)

    plt.legend(
        prop={'size': 18}
    )
    fig.savefig("figurer/mass_radius_comparison.pdf", bbox_inches="tight")
    

def plot_eos():
    pnr = np.linspace(0, .1, 1000)
    pur = np.linspace(0, 50, 1000)
    ps = [pnr, pur]
    fig, ax = plt.subplots(1, 2,figsize=(18, 6))
    up = get_u("fermi_gas_star/data/eos.npy")
    for i, p in enumerate(ps):
        u = [up(p0) for p0 in p]
        ax[i].plot(p, u, label="$ \\tilde u(\\tilde p)$")
        u = [u_fermi_nonrel(p0) for p0 in p]
        ax[i].plot(p, u, "k--", label="$ \\tilde u_{\mathrm{nr}} (\\tilde p)$")
        ax[i].plot(p, 3*p, "r-.", label="$ \\tilde u_{\mathrm{ur}} (\\tilde p)$")
        ax[i].set_xlabel("$p / p_0$")
        ax[i].set_ylabel("$u / u_0$")

    ax[0].legend(loc="upper left")

    fig.savefig("figurer/fermi_eos.pdf", bbox_inches="tight")
    

def plot_mass_of_pc():
    sols = load_sols()
    N = len(sols)    

    data = get_data(sols)
    data = np.array(data)
    _, M, pc = data
    y, x = M*m0, pc
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y)
    ax.set_xscale("log")

    n1 = 20 * 5
    x1, y1 = x[n1], y[n1]
    x2, y2 = x1*1e1, y1
    x3, y3 = x1*1e1, y1 - 0.1805
 
    ax.plot(x1, y1, "k.", ms=12)
    ax.plot(x2, y2, "k.", ms=12)
    ax.plot(x3, y3, "k.", ms=12)

    ax.arrow(x1*1.35, y1, (x2-x1)*0.6, (y2-y1), color="k",  width=0.001, head_width=0.01, head_length=3, length_includes_head=True)

    ax.text(x1, y1+0.015, "A")
    ax.text(x2, y2+0.015, "B")
    ax.text(x3, y3+0.015, "C")

    ax.set_ylabel("$M/M_\\odot$")
    ax.set_xlabel("$p_c/p_0$")


    fig.savefig("figurer/fermi_stability.pdf", bbox_inches="tight")    



plot_norm_pressure_mass()
# plot_mass_radius()
# plot_mass_radius_compare()
# plot_eos()
# plot_mass_of_pc()

# plot_mass_surface()
