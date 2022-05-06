import numpy as np
import matplotlib.pyplot as plt
import sys

from numpy import pi, sqrt, log10 as log
from matplotlib import cm, colors, collections

sys.path.append(sys.path[0] + "/..")
from chpt_numerics.chpt_eos import u_nr, u_ur, p, u, nI
from integrate_tov import get_u
from constants import get_const_pion, m_pi
from chpt_numerics.chpt_eos import alpha_0

from scipy.stats import linregress

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)

u0, m0, r0 = get_const_pion()


def load_sols(name=""):
    return np.load("pion_star/data/sols"+name+".npy", allow_pickle=True)


def get_data(sols):
    data = [[], [], []]
    for s in sols:
        data[0].append(s["R"])
        data[1].append(s["M"])
        data[2].append(s["pc"])
    return data


def plot_pressure_mass(name=""):
    all_sols = load_sols(name)
    sols = all_sols[30:180:8]
    N = len(sols)
    
    fig, ax = plt.subplots(2, figsize=(14, 10), sharex=True)
    [a.grid(linestyle="--", alpha=1, lw=0.4) for a in ax]
    p0s = []
    for i, s in enumerate(sols):
        p, m = s.y
        r = s.t
        R = r[-1]
        M = m[-1]
        p0 = p[0]
        p0s.append(p0)

        c = cm.viridis(i/N)

        ax[0].plot(r/R, p/p0, color=c, alpha=0.6)
        ax[1].plot(r/R, m/M, lw=2,  color=c, alpha=0.6)

    M = [s.y[1][-1] for s in all_sols]
    i = np.argmax(M)
    m = all_sols[i].y[1] / M[i]
    p = all_sols[i].y[0]
    p = p / p[0]
    r = all_sols[i].t
    r = r/r[-1]
    ax[0].plot(r, p, "--k", lw=2)
    ax[1].plot(r, m, "--k", lw=2, label="$M_\\mathrm{max}$") 

    ax[0].set_ylabel("$p/p_c$")
    ax[1].set_ylabel("$m/M$")
    ax[1].set_xlabel("$r / R$")

    c = np.arange(1, 5 + 1)

    norm = colors.Normalize(vmin=log(p0s[0]), vmax=log(p0s[-1]))
    cmap = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    cmap.set_array([])

    cb = fig.colorbar(cmap, ax=ax, location="right")
    cb.set_label( label="$\log_{10} [p_c / p_0] $", labelpad=25, rotation=270)

    fig.legend(bbox_to_anchor=(0.73, 0.87))

    fig.savefig("figurer/pion_star/pressure_mass_pion_star"+name+".pdf", bbox_inches="tight")



def plot_mass_radius_compare():
    sols1 = load_sols("")
    sols2 = load_sols("_non_rel")
    sols3 = load_sols("_newt")
    sols4 = load_sols("_newt_non_rel")
    sols = [sols1, sols2, sols3, sols4]

    N = len(sols1)
    assert N == len(sols2); assert N ==len(sols3)
    datas = [get_data(s) for s in sols]
        
    fig, ax = plt.subplots(figsize=(16, 8))

    linestyles = ["-", "-.", "--", ":"]
    labels = [
        "Full EOS + TOV",
        "Non-relativistic EOS + TOV",
        "Full EOS + Newtonian gravity",
        "Non-relativistic EOS + Newtonian gravity"
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
        lc = collections.LineCollection(segments, cmap='viridis', norm=norm, ls=linestyles[i], label=labels[i], lw=2)
        lc.set_array(z[::m])
        line = ax.add_collection(lc)

    cb = fig.colorbar(line)
    cb.set_label( label="$\log_{10} [p_c / p_0] $", labelpad=25, rotation=270)

    ax.set_xlabel("$R [\\mathrm{km}]$")
    ax.set_ylabel("$M / M_\odot$")
    ax.set_xlim(30, 100)
    ax.set_ylim(0, 15)

    plt.legend(prop={'size': 14}, loc=2)
    fig.savefig("figurer/pion_star/mass_radius_comparison.pdf", bbox_inches="tight")



def plot_gradient(x, y, z, ax, fig, zr=None, add_cb=True):
    if zr==None:
        zr = (z.min(), z.max())
    norm = colors.Normalize(*zr)
    points = np.array([x, y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
    lc = collections.LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(z)
    line = ax.add_collection(lc)
    if add_cb:
        cb = fig.colorbar(line, ax=ax)
        cb.set_label(label="$\log_{10} [p_c / p_0] $", labelpad=25, rotation=270)


def plot_mass_radius(name="", rmax=True):
    sols = load_sols(name)
    N = len(sols)
    data = get_data(sols)

    R, M, pc = [np.array(d) for d in data]
    x, y, z = R*r0, M*m0, log(pc)

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_gradient(x, y, z, ax, fig)

    Rs = np.linspace(0, 50, 100)
    ax.plot(Rs, 4 / 9 * Rs, "k--", label="$M = \\frac{4}{9} R$")

    i = np.argmax(M)
    label ="$(M, R) = " \
        + "(%.3f" %(M[i]*m0)\
        + "\, M_\odot, %.3f" %(R[i]*r0) \
        + "\, \mathrm{km})$ "
    ax.plot(R[i]*r0, M[i]*m0, "kx", ms=10, label=label) 

    if rmax:
        i = np.argmax(R)
        label ="$R_\\mathrm{max} = " + "{:.3f}".format(R[i]*r0) + "\, \mathrm{km}$"
        ax.plot(R[i]*r0, M[i]*m0, "ko", ms=10, label=label) 

    ax.set_xlim(8, 100)
    ax.set_ylim(0, 15)
    ax.set_xlabel("$R [\\mathrm{km}]$")
    ax.set_ylabel("$M / M_\odot$")
    plt.legend()
    
    fig.savefig("figurer/pion_star/mass_radius_pion_star" + name + ".pdf", bbox_inches="tight")
    
 

def plot_eos():
    fig, ax = plt.subplots(figsize=(10, 6))
    u = get_u("pion_star/data/eos.npy")

    pnr = np.linspace(0, .1, 1000)
    pur = np.linspace(0, 50, 1000)
    ps = [pnr, pur]
    fig, ax = plt.subplots(1, 2,figsize=(18, 6))
    for i, p in enumerate(ps):
        us1 = [u(p0) for p0 in p]
        us2 = [u_nr(p0) for p0 in p]
        us3 = u_ur(p)
        ax[i].plot(p, us1, label="$ \\tilde u(\\tilde p)$")
        ax[i].plot(p, us2, "k--", label="$ \\tilde u_\\mathrm{NR}(\\tilde p)$")
        ax[i].plot(p, us3, "r-.", label="$ \\tilde u_\\mathrm{UR}(\\tilde p)$")

        ax[i].set_xlabel("$p / p_0$")
        ax[i].set_ylabel("$u / u_0$")

    ax[0].legend(loc="upper left")

    fig.savefig("figurer/pion_star/pion_eos.pdf", bbox_inches="tight")


def plot_nlo_quantities():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    mu, alpha = np.load("pion_star/data/nlo_mu_alpha.npy")
    x = mu/m_pi
    ax.plot(x, alpha_0(x), "k--", label="$\\alpha_\\mathrm{LO}$")
    ax.plot(x, alpha, "r-.", label="$\\alpha_\\mathrm{NLO}$")

    ax.set_xlabel("$\\mu_I / m_\pi$")
    ax.set_ylabel("$\\alpha$")

    ax.set_ylim(-0.08, np.pi/2*1.02)
    ax.set_xlim(-0.1, 3)
    
    plt.legend()
    fig.savefig("figurer/pion_nlo_alpha.pdf", bbox_inches="tight")

    names = ["p", "u", "nI"]
    label = ["$p", "$u", "${n_{I,}}"]
    y_label = ["$p/p_0$", "$u/u_0$", "$n_I/n_0$"]
    j = np.where(x>=1)[0][0]

    lo_func = [p, u, nI]
    lo = [np.concatenate([np.zeros_like(x[:j]), f(1/x[j:])]) for f in lo_func]
    ns = [945, -1]
    for n in ns:
        for i, name in enumerate(names):
            y = np.load("pion_star/data/nlo_"+name+".npy")
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, lo[i], "k--", label=label[i]+"_\\mathrm{LO}$")
            ax.plot(x, y, "r-.", label=label[i]+"_\\mathrm{NLO}$")
            ax.set_ylim(-0.1, 1.05*y[n])
            ax.set_xlim(-0.1, x[n])
            ax.set_xlabel("$\\mu_I/m_\pi$")
            ax.set_ylabel(y_label[i])
            plt.legend()
            fig.savefig("figurer/pion_nlo_"+name+"_"+str(n)+".pdf", bbox_inches="tight")


def plot_eos_nlo():
    fig, ax = plt.subplots(figsize=(10, 6))
    p = np.linspace(0, 5, 1000)

    u = get_u("pion_star/data/eos.npy")
    ax.plot(p, u(p), "k--", label="$ u_{\\mathrm{LO}}(p)$")
    u = get_u("pion_star/data/eos_nlo.npy")
    ax.plot(p, u(p), "r-.", label="$ u_\\mathrm{NLO}(p)$")
    ax.set_xlabel("$p / p_0$")
    ax.set_ylabel("$u / u_0$")

    ax.legend(loc="upper left")
    fig.savefig("figurer/pion_eos_nlo.pdf", bbox_inches="tight")


def plot_eos_leptons():
    fig, ax = plt.subplots(figsize=(8, 6))
    u = get_u("pion_star/data/eos.npy")
    u_e = get_u("pion_star/data/eos_e.npy")
    u_mu = get_u("pion_star/data/eos_mu.npy")

    p = np.linspace(0, 0.01, 1000) 
    fig, ax = plt.subplots(figsize=(12, 6)) 

    ax.plot(p, [u(p0) for p0 in p], label="$ \\pi$")
    ax.plot(p, [u_e(p0) for p0 in p], "k--", label="$ \\pi + e$")
    ax.plot(p, [u_mu(p0) for p0 in p], "r-.", label="$ \\pi + \\mu$")

    ax.set_xlabel("$p / u_0$")
    ax.set_ylabel("$u / u_0$")

    ax.legend()
    fig.savefig("figurer/pion_star/pion_eos_leptson.pdf", bbox_inches="tight")


def plot_all_eos():
    fig, ax = plt.subplots(figsize=(12, 6))

    p = np.linspace(0, 0.8, 1000)

    names = ["", "_e", "_mu", "_neutrino"]
    labels = ["$\\pi$", "$\\pi+e$", "$\\pi+\\mu$", "$\\pi+\\ell+\\nu_\\ell$"]
    lines = ["-", "-.", "--", ":"]
    # colors = ["tab:blue", "green", "red", "black"]
    colors = ["blue", "green", "orange", "black"]

    for i, name in enumerate(names):
        u = get_u("pion_star/data/eos"+name+".npy")
        ax.plot(p, [u(p0) for p0 in p], lines[i], color=colors[i], label=labels[i], lw=2, alpha=0.8)

    ax.set_xlabel("$p / u_0$")
    ax.set_ylabel("$u / u_0$")
    ax.legend()
    ax.set_ylim(-0.1, 2.6)
    fig.savefig("figurer/pion_star/pion_all_eos.pdf", bbox_inches="tight")


def plot_mu():
    fig, ax = plt.subplots(figsize=(12, 6))
    p = lambda x: 1/2 * (x**2 + 1/x**2 - 2)
    mu = np.linspace(1, 10, 100)
    ax.plot(mu, p(mu), "r-.", label="$ \\tilde p (\\mu_I/m_\\pi)$")
    ax.set_xlabel("$\\mu / m_\\pi$")
    ax.set_ylabel("$p / p_0$")

    ax.legend()

    fig.savefig("figurer/pion_star/pion_mu.pdf", bbox_inches="tight")


def plot_eos_EM():
    p = np.linspace(0, 0.6, 1000)
    fig, ax = plt.subplots(figsize=(10, 6))

    u = get_u("pion_star/data/eos.npy")
    us = [u(p0) for p0 in p]
    ax.plot(p, us, label="$ \\tilde u(\\tilde p)$", lw=1, alpha=0.8)

    u = get_u("pion_star/data/eos_EM.npy")
    us = [u(p0) for p0 in p]
    ax.plot(p, us, "k--", label="$ \\tilde u_{\\mathrm{EM}}(\\tilde p)$")

    ax.set_xlabel("$p / p_0$")
    ax.set_ylabel("$u / u_0$")
    ax.legend(loc="upper left")

    fig.savefig("figurer/pion_star/pion_eos_EM.pdf", bbox_inches="tight")


def plot_u_p():
    from pion_star_eos import u, p, uEM, pEM, D
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    N = 200
    y0 = np.linspace(0, 1.2, N)

    yc = 1
    x = 1/(y0 + yc)

    ax[0].plot(y0, p(x), label="$p(\\mu_I)$")
    ax[1].plot(y0, u(x), label="$u(\\mu_I)$")
    
    yc = sqrt(1 + 2*D)
    x = 1/(yc + y0)

    ax[0].plot(y0, pEM(x, D), "k--", label="$p_{\\mathrm{EM}}(\\mu_I)$")
    ax[1].plot(y0, uEM(x, D), "k--", label="$u_{\\mathrm{EM}}(\\mu_I)$")
 
    [a.legend() for a in ax]

    [a.set_xlabel("$(\\mu_I - \\mu_I^c)/\\bar m$") for a in ax]
    ax[0].set_ylabel("$p/p_0$")
    ax[1].set_ylabel("$u/u_0$")

    fig.savefig("figurer/pion_star/pion_up.pdf", bbox_inches="tight")


def plot_mass_radius_compare_EM():
    sols1 = load_sols()
    sols2 = load_sols(name="_EM")
    N = len(sols1)
    sols = [sols1, sols2]
    datas = [get_data(s) for s in sols]
    fig, ax = plt.subplots(figsize=(16, 8))

    labels = ["Only strong interactions", "EM interactions"]
    colors = ["tab:blue", "k"]
    alpha = [1, 0.8]
    style = ["-","--"]
    marker = ["x", "*"]
    for i, data in enumerate(datas):
        R, M, pc = [np.array(d) for d in data]
        x, y, z = R*r0, M*m0, log(pc)

        ax.plot(x, y, label=labels[i], color=colors[i], ls=style[i], alpha=alpha[i])
        j = np.argmax(M)
        label ="$(M, R) = " \
            + "(%.3f" %(M[j]*m0)\
            + "\, M_\odot, %.3f" %(R[j]*r0) \
            + "\, \mathrm{km})$ "
        ax.plot(R[j]*r0, M[j]*m0, "k", marker=marker[i], ls="", ms=10, label=label)

    ax.set_xlabel("$R [\\mathrm{km}]$")
    ax.set_ylabel("$M / M_\odot$")
    plt.legend()
    
    fig.savefig("figurer/pion_star/mass_radius_pion_star_compare.pdf", bbox_inches="tight")
    

def plot_lepton_compare():
    sols1 = load_sols()
    sols2 = load_sols(name="_e")
    sols3 = load_sols(name="_mu")
    N = len(sols1)
    sols = [sols1, sols2, sols3]
    datas = [get_data(s) for s in sols]

    u0, m0, r0 = get_const_pion()

    fig, ax = plt.subplots(figsize=(16, 10))
    lines = ["-", "--", "-."]
    colors = ["tab:blue", "tab:green", "r"]
    labels = ["\\pi", "\\pi+e", "\\pi+\\mu"]
    marker=["", "x", "*"]
    for i, data in enumerate(datas):
        R, M, pc = [np.array(d) for d in data] 
        ax.plot(R*r0, M*m0, ls=lines[i], color=colors[i], label="$"+labels[i]+"$")
    
    ax.set_xlabel("$R [\\mathrm{km}]$")
    ax.set_ylabel("$M / M_\odot$")
    ax.set_xscale("log")
    ax.set_yscale("log")

    Rs = np.linspace(0, 8e4, 100)
    ax.set_ylim(3e-2, 1e3)
    ax.plot(Rs, 4 / 9 * Rs, "k--", label="$M = \\frac{4}{9} R$")
    ax.legend(loc="lower right")

    fig.savefig("figurer/pion_star/mass_radius_lepton_compare.pdf", bbox_inches="tight")


def plot_all():
    sols1 = load_sols()
    sols2 = load_sols(name="_e")
    sols3 = load_sols(name="_mu")
    sols4 = load_sols(name="_neutrino")
    N = len(sols1)
    sols = [sols1, sols2, sols3, sols4]
    datas = [get_data(s) for s in sols]

    u0, m0, r0 = get_const_pion()

    fig, ax = plt.subplots(figsize=(16, 10))
    lines = ["-", "--", "-.", ":"]
    # colors = ["blue", "green", "r", "purple"]
    colors = ["blue", "green", "orange", "black"]
    labels = ["\\pi", "\\pi+e", "\\pi+\\mu", "\\pi+\\ell+\\nu_\\ell"]
    for i, data in enumerate(datas):
        R, M, pc = [np.array(d) for d in data] 
        ax.plot(R*r0, M*m0, ls=lines[i], color=colors[i], label="$"+labels[i]+"$", lw=3, alpha=0.65)
    
    ax.set_xlabel("$R [\\mathrm{km}]$")
    ax.set_ylabel("$M / M_\odot$")
    ax.set_xscale("log")
    ax.set_yscale("log")

    Rs = np.linspace(0, 8e4, 100)
    ax.set_ylim(3e-2, 1e3)
    ax.plot(Rs, 4 / 9 * Rs, "k--", label="$M = \\frac{4}{9} R$")
    ax.legend(loc="lower right")

    fig.savefig("figurer/pion_star/mass_radius_all.pdf", bbox_inches="tight")


def plot_lepton(name = "_e"):
    sols = load_sols(name=name)
    N = len(sols)
    data = get_data(sols)
    u0, m0, r0 = get_const_pion()

    fig, ax = plt.subplots(figsize=(16, 6))
    R, M, pc = [np.array(d) for d in data]

    x, y, z = R*r0, M*m0, log(pc)
    plot_gradient(x, y, z, ax, fig)

    j = np.argmax(M)
    label ="$(M, R) = " \
        + "(%.3f" %(M[j]*m0)\
        + "\, M_\odot, %.3f" %(R[j]*r0) \
        + "\, \mathrm{km})$ "
    ax.plot(R[j]*r0, M[j]*m0, "kx", label=label)

    pc_max = data[2][j]
    
    ax.set_xlabel("$R [\\mathrm{km}]$")
    ax.set_ylabel("$M / M_\odot$")

    if name == "_e":
        ax.set_xlim(0, 4e5)
    else: 
        ax.set_xlim(0, 2e3)

    ax.ticklabel_format(style="scientific", scilimits=(-2, 2))
    ax.legend()

    fig.savefig("figurer/pion_star/mass_radius_"+name+".pdf", bbox_inches="tight")


def plot_neutrino(name = "_neutrino"):
    sols = load_sols(name=name)
    N = len(sols)
    data = get_data(sols)
    
    u0, m0, r0 = get_const_pion()

    fig, ax = plt.subplots(figsize=(16, 6))
    R, M, pc = [np.array(d) for d in data] 
    x, y, z = R*r0, M*m0, log(pc)
    plot_gradient(x, y, z, ax, fig)

    j = np.argmax(M)
    label ="$(M, R) = " \
        + "(%.3f" %(M[j]*m0)\
        + "\, M_\odot, %.3f" %(R[j]*r0) \
        + "\, \mathrm{km})$ "
    ax.plot(R[j]*r0, M[j]*m0, "kx", label=label)
    
    ax.set_xlabel("$R [\\mathrm{km}]$")
    ax.set_ylabel("$M / M_\odot$")

    ax.ticklabel_format(style="scientific", scilimits=(-2, 2))
    ax.legend()

    fig.savefig("figurer/pion_star/mass_radius_neutrino.pdf", bbox_inches="tight")
    
    
def plot_light():

    fig, ax = plt.subplots(figsize=(16, 6))
    pmins = [0.1, 0.01, 0.001]
    names = ["_light_%.2e"%pmin for pmin in pmins]
    names.append("_neutrino")

    for name in names:
        sols = load_sols(name=name)
        N = len(sols)
        data = get_data(sols)

        u0, m0, r0 = get_const_pion()
        R, M, pc = [np.array(d) for d in data]
        x, y, z = R*r0, M*m0, log(pc)
        ax.loglog(x[1::], y[1::])
    
    ax.set_xlabel("$R [\\mathrm{km}]$")
    ax.set_ylabel("$M / M_\odot$")

    fig.savefig("figurer/pion_star/mass_radius_light.pdf", bbox_inches="tight")


def plot_nlo():
    fig, ax = plt.subplots(figsize=(16, 6))
    sol1 = load_sols()
    sol2 = load_sols(name="_nlo")
    sols = [
        sol1,
        sol2
    ]
    u0, m0, r0 = get_const_pion()
    label = ["LO", "NLO"]
    line = ["k--", "r-."]
    for i, sol in enumerate(sols):
        data = get_data(sol)

        R, M, pc = [np.array(d) for d in data]
        imin = np.where(pc>1e-5)[0][0]
        imax = np.where(pc>2.9e1)[0][0]
        x, y, z = R*r0, M*m0, log(pc)
        x, y, z = x[imin:imax], y[imin:imax], z[imin:imax]
        ax.plot(x, y, line[i], label=label[i])

    ax.set_xlabel("$R [\\mathrm{km}]$")
    ax.set_ylabel("$M / M_\odot$")
    ax.set_title("$p_c/p_0 \\in [10^{-5}, 30]$")

    plt.legend()
    fig.savefig("figurer/pion_star/mass_compare_order.pdf", bbox_inches="tight")

def test():

    fig, ax = plt.subplots(figsize=(16, 6))
    pmins = [0.1, 0.01, 0.001]
    names = ["_light_%.2e"%pmin for pmin in pmins]
    names.append("_neutrino")

    for name in names:
        sols = load_sols(name=name)
        N = len(sols)
        data = get_data(sols)

        u0, m0, r0 = get_const_pion()
        R, M, pc = [np.array(d) for d in data]
        x, y, z = R*r0, M*m0, log(pc)
        ax.loglog(x[1::], y[1::])
    
    ax.set_xlabel("$R [\\mathrm{km}]$")
    ax.set_ylabel("$M / M_\odot$")

    fig.savefig("figurer/pion_star/mass_radius_light.pdf", bbox_inches="tight")


# plot_pressure_mass()
# plot_pressure_mass(name="_EM")

# plot_mass_radius()
# plot_mass_radius_compare()
# plot_mass_radius(name="_EM")
# plot_mass_radius_compare_EM()

# plot_eos()
# plot_mu()
# plot_eos_EM()
# plot_u_p()

# plot_lepton()
# plot_lepton(name="_mu")
# plot_lepton_compare()

# plot_all_eos()
# plot_neutrino()

# plot_all()
# test()

# plot_nlo_quantities()
# plot_eos_nlo()

# plot_mass_radius("_nlo", rmax=False)
plot_nlo()
