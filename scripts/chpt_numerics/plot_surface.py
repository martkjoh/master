import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

sys.path.append(sys.path[0] + "/..")
from spectrum import *
from free_energy_nlo import F_0_2
from constants import m_S, m_pi, f_pi

plt.rc("font", family="serif", size=16)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)

lo = lambda x: x.subs(m, m_pi).subs(f, f_pi).subs(mS, m_S)
num_lo = lambda x: lambdify((muI, a), lo(x), "numpy")


def plot_free_energy_surface():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 6))

    N = 30
    d = 0.6
    alpha = np.linspace(-d, np.pi + d, N)
    mu = np.linspace(0, 2.5, N)*m_pi
    a_lo_list = alpha_0(mu/m_pi)
    MU, A = np.meshgrid(mu, alpha)
    F = num_lo(F_0_2)

    FLO = F(MU, A)

    ax.plot(mu[:-1]/m_pi, a_lo_list[:-1], F(mu, a_lo_list)[:-1]+0.04, "k--", zorder=10)
    ax.plot(mu/m_pi, a_lo_list, np.min(FLO)+0.1, "k--")
 
    surf = ax.plot_surface(MU/m_pi, A, FLO, cmap="viridis", alpha=0.7)
    surf = ax.plot_wireframe(MU/m_pi, A, FLO, color="black", lw=0.15, alpha=0.6)

    ax.azim=-35
    ax.elev=25

    plt.xlabel("$\\mu_I/m_\\pi$")
    ax.set_ylabel("$\\alpha$")
    ax.set_zlabel("$\\mathcal{F}/u_0$")
    ax.zaxis.set_tick_params(labelsize=10)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


    plt.subplots_adjust(top=1, bottom=0, right=0.8, left=0, hspace=0, wspace=1)
    save_opt = dict(
        bbox_inches='tight',
        pad_inches = 0, 
        transparent=True, 
        dpi=300
    )


    ax.set_zticks(np.arange(-16, -11, 1.))
    ax.set_zlim(-16.1,-11.9)
    plt.savefig("figurer/free_energy_surface.pdf", **save_opt)


def plot_free_energy_surface_wo_axis():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 8))

    N = 30
    d = 0.6
    alpha = np.linspace(-d, np.pi + d, N)
    mu = np.linspace(0, 2.5, N)*m_pi
    a_lo_list = alpha_0(mu/m_pi)
    MU, A = np.meshgrid(mu, alpha)
    F = num_lo(F_0_2)

    FLO = F(MU, A)

 
    surf = ax.plot_surface(MU/m_pi, A, FLO, cmap="viridis", alpha=0.7)
    surf = ax.plot_wireframe(MU/m_pi, A, FLO, color="black", lw=0.15, alpha=0.6)


    ax.plot(mu/m_pi, a_lo_list, F(mu, a_lo_list)+0.01, "k--", lw=2, alpha=1, zorder=10)


    ax.azim=-35
    ax.elev=25

    # Hide grid lines
    ax.grid(False)
    ax.axis("off")

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    save_opt = dict(
        bbox_inches='tight',
        pad_inches = -0.4,
        transparent = True, 
        dpi=300
    )

    ax.set_xlim([0,2.5])
    ax.set_ylim([-0.5,4])
    ax.set_zlim([-16.,-12.2])

    fig.savefig("figurer/free_energy_surface_wo_axis.pdf", **save_opt)


plot_free_energy_surface()
plot_free_energy_surface_wo_axis()
