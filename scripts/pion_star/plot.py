import numpy as np
import matplotlib.pyplot as plt
import sys

from numpy import pi, sqrt, log10 as log
from matplotlib import cm, colors, collections

from chpt_eos import u_nr, u_ur, p
sys.path.append(sys.path[0] + "/..")
from integrate_tov import get_u
from constants import get_const_pion

from scipy.stats import linregress

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)

u0, m0, r0 = get_const_pion()


def load_sols(name=""):
    return np.load("pion_star/data/sols"+name+".npy", allow_pickle=True)


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
    datas = [[[], [], []] for _ in sols]
    for j, solsi in enumerate(sols):
        for i, s in enumerate(solsi):
            datas[j][0].append(s.t[-1])
            datas[j][1].append(s.y[1][-1])
            datas[j][2].append(s.y[0][0])
        
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



def plot_mass_radius(name=""):
    sols = load_sols(name)
    N = len(sols)    

    data = [[], [], []]
    for i, s in enumerate(sols):
        data[0].append(s.t[-1])
        data[1].append(s.y[1][-1])
        data[2].append(s.y[0][0])


    R, M, pc = [np.array(d) for d in data]
    x, y, z = R*r0, M*m0, log(pc)

    fig, ax = plt.subplots(figsize=(16, 8))

    norm = colors.Normalize(z.min(), z.max()) 
    points = np.array([x, y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
    lc = collections.LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(z)
    line = ax.add_collection(lc)
    cb = fig.colorbar(line, ax=ax)
    cb.set_label(label="$\log_{10} [p_c / p_0] $", labelpad=25, rotation=270)


    Rs = np.linspace(0, 50, 100)
    ax.plot(Rs, 4 / 9 * Rs, "k--", label="$M = \\frac{4}{9} R$")

    i = np.argmax(M)
    label ="$(M, R) = " \
        + "(%.3f" %(M[i]*m0)\
        + "\, M_\odot, %.3f" %(R[i]*r0) \
        + "\, \mathrm{km})$ "
    ax.plot(R[i]*r0, M[i]*m0, "kx", ms=10, label=label) 

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
    datas = [[[], [], []] for _ in sols]
    for j, sol in enumerate(sols):
        for i, s in enumerate(sol):
            datas[j][0].append(s.t[-1])
            datas[j][1].append(s.y[1][-1])
            datas[j][2].append(s.y[0][0])

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
    


def test():
    sols1 = load_sols()
    sols2 = load_sols(name="_e")
    sols3 = load_sols(name="_mu")
    N = len(sols1)
    sols = [sols1, sols2, sols3]
    datas = [[[], [], []] for _ in sols]
    for j, sol in enumerate(sols):
        for i, s in enumerate(sol):
            datas[j][0].append(s.t[-1])
            datas[j][1].append(s.y[1][-1])
            datas[j][2].append(s.y[0][0])


    u0, m0, r0 = get_const_pion()

    fig, ax = plt.subplots(figsize=(16, 6))
    lines = ["-", "--", "-."]
    colors = ["tab:blue", "k", "r"]
    labels = ["\\pi", "\\pi+e", "\\pi+\\mu"]
    for i, data in enumerate(datas):
        R, M, pc = [np.array(d) for d in data] 
        ax.plot(R*r0, M*m0, ls=lines[i], color=colors[i], label="$"+labels[i]+"$")
 
    ax.set_xlabel("$R [\\mathrm{km}]$")
    ax.set_ylabel("$M / M_\odot$")
    ax.legend()

    fig.savefig("figurer/pion_star/mass_radius_lepton.pdf", bbox_inches="tight")



test()


# test()

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
