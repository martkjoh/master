import numpy as np
import matplotlib.pyplot as plt
import sys

from numpy import pi, sqrt, log10 as log
from matplotlib import cm, colors, collections

sys.path.append(sys.path[0] + "/..")
from chpt_numerics.chpt_eos import u_nr, u_ur, p, u, nI
from integrate_tov import get_u
from constants import get_const_pion, m_pi, m_e
from chpt_numerics.chpt_eos import alpha_0

from scipy.stats import linregress

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)

u0, m0, r0 = get_const_pion()

# linetypes
dashdotdot = (0, (6, 1, 1, 1, 1, 1))
dashdashdot = (0, (6, 1, 6, 1, 1, 1))

def load_sols(name=""):
    return np.load("pion_star/data/sols"+name+".npy", allow_pickle=True)

def load_max(name=""):
    return np.load("pion_star/data/max"+name+".npy", allow_pickle=True)[None][0]


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
    pcs = []
    for i, s in enumerate(sols):
        f = s["f"]
        R, M, pc = s["R"], s["M"], s["pc"]
        pcs.append(pc)

        c = cm.viridis(i/(N+1))
        r = np.linspace(0, R, 200)

        ax[0].plot(r/R, f(r)[0]/pc, color=c, alpha=0.6)
        ax[1].plot(r/R, f(r)[1]/M, lw=2,  color=c, alpha=0.6)

    i = np.argmax(np.array([s["M"] for s in all_sols]))
    M, R, pc = [all_sols[i][s] for s in ["M", "R", "pc"]]
    r = np.linspace(0, R)
    f = all_sols[i]["f"]
    p, m = f(r)

    ax[0].plot(r/R, p/pc, "--k", lw=2)
    ax[1].plot(r/R, m/M, "--k", lw=2, label="$M_\\mathrm{max}$") 

    ax[0].set_ylabel("$p/p_c$")
    ax[1].set_ylabel("$m/M$")
    ax[1].set_xlabel("$r / R$")
    fig.legend(bbox_to_anchor=(0.73, 0.87))


    norm = colors.Normalize(vmin=log(pcs[0]), vmax=log(pcs[-1]))
    cmap = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    cb = fig.colorbar(cmap, ax=ax, location="right")
    cb.set_label( label="$\log_{10} [p_c / p_0] $", labelpad=25, rotation=270)

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



def plot_nlo_quantities_lattice():
    mu, alpha = np.load("pion_star/data/nlo_mu_alpha.npy")
    x1 = mu/m_pi
    mu, alpha = np.load("pion_star/data/nlo_mu_alphalattice.npy")
    x2 = mu/131

    
    names = ["p", "u", "nI"]
    label = ["$p", "$u", "${n_{I,}}"]
    y_label = ["$p/p_0$", "$u/u_0$", "$n_I/n_0$"]
    n=-1
    for i, name in enumerate(names):
        y = np.load("pion_star/data/nlo_"+name+".npy")
        z = np.load("pion_star/data/nlo_"+name+"lattice.npy")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x1, z, "k--", label=label[i]+"_\\mathrm{Lattice}$")
        ax.plot(x2, y, "r-.", label=label[i]+"_\\mathrm{PDG}$")
        ax.set_xlabel("$\\mu_I/m_\pi$")
        ax.set_ylabel(y_label[i])
        plt.legend()
        fig.savefig("figurer/lattice_nlo_"+name+"_"+str(n)+".pdf", bbox_inches="tight")



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

def plot_eos_nlo_lattice():
    fig, ax = plt.subplots(figsize=(10, 6))
    p = np.linspace(0, 5, 1000)

    u = get_u("pion_star/data/eos_nlolattice.npy")
    ax.plot(p, u(p), "k--", label="$ u_{\\mathrm{lattice}}(p)$")
    u = get_u("pion_star/data/eos_nlo.npy")
    ax.plot(p, u(p), "r-.", label="$ u_\\mathrm{NLO}(p)$")
    ax.set_xlabel("$p / p_0$")
    ax.set_ylabel("$u / u_0$")

    ax.legend(loc="upper left")
    fig.savefig("figurer/lattice_pion_eos_nlo.pdf", bbox_inches="tight")


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


def plot_neutrino_nlo_eos():
    fig, ax = plt.subplots(figsize=(12, 6))

    p = np.linspace(0, 5, 1000)


    names = ["_neutrino", "_neutrino_nlo"]
    labels = ["$\\pi\\ell\\nu_\\ell \, \\mathrm{LO}$", "$\\pi\\ell\\nu_\\ell  \, \\mathrm{NLO}$"]
    lines = ["--", "-."]
    colors = ["black", "red"]

    for i, name in enumerate(names):
        u = get_u("pion_star/data/eos"+name+".npy")
        ax.plot(p, u(p), lines[i], color=colors[i], label=labels[i], lw=2, alpha=0.8)

    ax.set_xlabel("$p / u_0$")
    ax.set_ylabel("$u / u_0$")
    ax.legend()

    fig.savefig("figurer/pion_star/neutrino_nlo_eos.pdf", bbox_inches="tight")


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
    from chpt_numerics.chpt_eos import u, p, uEM, pEM, D
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    N = 200
    y = np.linspace(0, 1.8, N)
    x1 = 1 / (y+1)
    x2 = 1 / (y+sqrt(1+D))

    ax[0].plot(y, p(x1), label="$p(\\mu_I)$")
    ax[1].plot(y, u(x1), label="$u(\\mu_I)$")

    ax[0].plot(y, pEM(x2, D), "k--", label="$p_{\\mathrm{EM}}(\\mu_I)$")
    ax[1].plot(y, uEM(x2, D), "k--", label="$u_{\\mathrm{EM}}(\\mu_I)$")
 
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


def plot_mass_radius_lattice():
    sols = load_sols()
    N = len(sols)
    sols = [sols, sols]
    datas = [get_data(s) for s in sols]
    fig, ax = plt.subplots(figsize=(16, 8))

    labels = ["PDG", " Lattice"]
    colors = ["tab:blue", "k"]
    alpha = [1, 0.8]
    style = ["-","--"]
    marker = ["x", "*"]
    lattice = [False, True]
    for i, data in enumerate(datas):

        if lattice[i]: from constants_lattice import get_const_pion
        else: from constants import get_const_pion
        u0, m0, r0 = get_const_pion()

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
    fig.savefig("figurer/pion_star/lattice_mass_radius_pion_star_compare.pdf", bbox_inches="tight")
    

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
    sols1 = load_sols(name="_nlo")
    sols2 = load_sols(name="_e")
    sols3 = load_sols(name="_mu")
    sols4 = load_sols(name="_neutrino")
    sols = [sols1, sols2, sols3, sols4]
    datas = [get_data(s) for s in sols]

    u0, m0, r0 = get_const_pion()

    fig, ax = plt.subplots(figsize=(16, 8))
    lines = [dashdotdot, "--", "-.", "-"]
    N = len(datas)
    colors = [cm.winter(i/(N-1)) for i in range(N)]
    labels = ["\\pi\,\\mathrm{NLO}", "\\pi e", "\\pi\\mu", "\\pi\\ell\\nu_\\ell"]
    for i, data in enumerate(datas):
        R, M, pc = [np.array(d) for d in data] 
        ax.plot(R*r0, M*m0, ls=lines[i], color=colors[i], label="$"+labels[i]+"$", lw=3, alpha=0.65)
    
    ax.set_xlabel("$R [\\mathrm{km}]$")
    ax.set_ylabel("$M / M_\odot$")
    ax.set_xscale("log")
    ax.set_yscale("log")

    Rs = np.linspace(0, 8e4, 100)
    ax.set_ylim(3e-2, 3e2)
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


def plot_nlo(t=""):
    fig, ax = plt.subplots(figsize=(16, 6))
    sol1 = load_sols(name=t)
    sol2 = load_sols(name=t+"_nlo")
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
    
    plt.legend()
    fig.savefig("figurer/pion_star/mass_compare_order"+t+".pdf", bbox_inches="tight")


def plot_phase():
    F = lambda mu, a: -1/2 * (2*np.cos(a) + mu**2*np.sin(a)**2)
    g = lambda mu, a: 1/2*(1 - mu**2)*a**2 + 1/24*(4*mu**2 - 1)*a**4
    N = 30
    a = np.linspace(-0.05, 0.8, N)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    F0 = F(0, 0)

    F1,F2,F3 = F(0.9, a)-F0, F(1., a)-F0, F(1.1, a)-F0

    i = np.argmin(F3)
    d = .004
    l = np.array([[0, d], [a[i], F3[i]+d]]).T
 
    ax.plot(a, F1, "royalblue", label=r"$\mu_I<m_\pi$")
    ax.plot(a, F2, "k--", label=r"$\mu_I=m_\pi$")
    ax.plot(a, F3, "k", label=r"$\mu_I>m_\pi$")


    ax.scatter(l[0], l[1], s=400)

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$(\mathcal{F} - \mathcal{F}_0)/u_0$")

    plt.tight_layout()
    plt.legend()
    plt.savefig("figurer/phase_transition.pdf")


def plot_light_log():
    fig, ax = plt.subplots(figsize=(16, 6))

    pmins = [0.1, 10**(-1.5), (1+m_e/m_pi) / (12*pi**2), 10**(-2.5), 0.001]

    u0, m0, r0 = get_const_pion()

    names = ["_light_%.2e"%pmin for pmin in pmins]
    for i, name in enumerate(names):
        sols = load_sols(name=name)
        N = len(sols)
        data = get_data(sols)

        R, M, pc = [np.array(d) for d in data]
        x, y, z = R*r0, M*m0, log(pc)
        color = cm.viridis(i/(len(names)+1))
        label = "$p_\\mathrm{min}=%.1e$"%pmins[i]
        ax.loglog(x, y, label=label, color=color)

    sols = load_sols(name="_neutrino")
    data = get_data(sols)

    R, M, pc = [np.array(d) for d in data]
    x, y, z = R*r0, M*m0, log(pc)
    label = "$\\pi\\ell\\nu_\\ell$"
    ax.loglog(x, y, "k--", label=label)
    
    ax.set_xlabel("$R [\\mathrm{km}]$")
    ax.set_ylabel("$M / M_\odot$")
    plt.legend()
    fig.savefig("figurer/pion_star/mass_radius_light_log.pdf", bbox_inches="tight")


def plot_light():
    fig, ax = plt.subplots(figsize=(16, 6))


    p0 = (1+m_e/m_pi) / (12*pi**2)
    pmins = np.logspace( np.log10(p0/3), np.log10(p0*3), 5 )

    u0, m0, r0 = get_const_pion()

    x = 2
    b = np.log10(pmins[0]*.8)
    a = np.log10(pmins[-1]*1.2)
    d = b - a

    names = ["_light_%.2e"%pmin for pmin in pmins]
    for i, name in enumerate(names):
        sols = load_sols(name=name)
        N = len(sols)
        data = get_data(sols)

        R, M, pc = [np.array(d) for d in data]
        x, y, z = R*r0, M*m0, log(pc)
        color = cm.viridis( (np.log10(pmins[i]) - a) / (b - a) )
        label = "$p_\\mathrm{min}=%.1e$"%pmins[i]
        ax.plot(x, y, color=color)

    sols = load_sols(name="_neutrino")
    data = get_data(sols)

    R, M, pc = [np.array(d) for d in data]
    x, y, z = R*r0, M*m0, log(pc)
    label = "$\\pi\\ell\\nu_\\ell$"
    ax.plot(x, y, "k--", label=label)
    
    ax.set_xlabel("$R [\\mathrm{km}]$")
    ax.set_ylabel("$M / M_\odot$")
    plt.legend()
     
    norm = colors.Normalize(vmin=a, vmax=b)
    cmap = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    cb = fig.colorbar(cmap, ax=ax, location="right")
    cb.set_label( label="$\log_{10} [p_\\mathrm{min} / p_0] $", labelpad=25, rotation=270)

    fig.savefig("figurer/pion_star/mass_radius_light.pdf", bbox_inches="tight")


def plot_light_nogrid():
    fig, ax = plt.subplots(figsize=(16, 6))

    p0 = (1+m_e/m_pi) / (12*pi**2)
    pmins = np.logspace( np.log10(p0/3), np.log10(p0*3), 5 )

    u0, m0, r0 = get_const_pion()

    x = 2
    b = np.log10(pmins[0]*.8)
    a = np.log10(pmins[-1]*1.2)
    d = b - a

    names = ["_light_%.2e"%pmin for pmin in pmins]
    for i, name in enumerate(names):
        sols = load_sols(name=name)
        N = len(sols)
        data = get_data(sols)

        R, M, pc = [np.array(d) for d in data]
        x, y, z = R*r0, M*m0, log(pc)
        color = cm.viridis( (np.log10(pmins[i]) - a) / (b - a) )
        label = "$p_\\mathrm{min}=%.1e$"%pmins[i]
        ax.plot(x[1:], y[1:], color=color)

    ax.set_axis_off()

    fig.savefig("figurer/pion_star/mass_radius_light_nogrid.pdf", bbox_inches="tight")



def plot_max():
    names = ["", "_nlo", "_EM", "_e", "_mu", "_neutrino"]
    labels = ["$\\pi\,\mathrm{LO}$", "$\\pi\,\mathrm{NLO}$", "$\\pi\,\mathrm{EM}$", "$\\pi e$", "$\\pi \\mu$", "$\\pi \\ell\\nu_\\ell$"]

    names = ["_nlo", "_e", "_mu", "_neutrino"]
    labels = ["$\\pi\,\mathrm{NLO}$", "$\\pi e$", "$\\pi \\mu$", "$\\pi \\ell\\nu_\\ell$"]

    N = len(names)
    sols = [load_max(name) for name in names]
    colors = [cm.winter(i/(N-1)) for i in range(N)]
    ls = [dashdotdot, "-", "--", "-."]
    fig, ax = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    for i, name in enumerate(names):
        s = sols[i]
        f = s["f"]
        R, M, pc = s["R"], s["M"], s["pc"]
        r = np.linspace(0, R, 200)
        p, m = f(r)

        ax[1].plot(r/R, p/pc, color=colors[i], ls=ls[i], label=labels[i])
        ax[0].plot(r/R, m/M, color=colors[i], ls=ls[i], label=labels[i])

    ax[0].legend()
    ax[1].set_ylabel("$p/p_c$")
    ax[0].set_ylabel("$m/M$")
    ax[1].set_xlabel("$r/R$")

    fig.savefig("figurer/pion_star/max_pressure_mass.pdf", bbox_inches="tight")


if __name__=="__main__":
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
    # plot_neutrino_nlo_eos()

    # plot_all()

    # plot_nlo_quantities()
    # plot_eos_nlo()

    # plot_mass_radius("_nlo", rmax=False)
    # plot_nlo()
    # plot_nlo("_neutrino")

    plot_light()
    plot_light_nogrid()
    # plot_light_log()
    # plot_max()
    # plot_phase()

    # ### Plots using lattice constants

    # plot_nlo_quantities_lattice()
    # plot_eos_nlo_lattice()
    # plot_nlo_lattice()
    # plot_u_p()
