import numpy as np
from numpy import pi, sqrt, arcsinh
from scipy.optimize import newton
from matplotlib import pyplot as plt
import sys

sys.path.append(sys.path[0] + "/..")
from constants import get_const_lepton, f_pi, m_e, m_mu, m_pi
from integrate_tov import get_u

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)


xf = lambda x: sqrt(x**2 - 1)
ul = lambda x: (2*xf(x)**3 + xf(x)) * sqrt(1 + xf(x)**2) - arcsinh(xf(x)) 
pl = lambda x: 1 / 3 * ((2*xf(x)**3 - 3*xf(x)) * sqrt(1 + xf(x)**2) + 3*arcsinh(xf(x)))

ul = lambda x: (2*x**3 + x) * sqrt(1 + x**2) - arcsinh(x) 
pl = lambda x: 1 / 3 * ((2*x**3 - 3*x) * sqrt(1 + x**2) + 3*arcsinh(x))


p_pi = lambda  x: 1/2 * (x - 1/x)**2
u_pi = lambda x: 1/2 * (2 + x**2 - 3/x**2)

p_pi_lim = lambda x: 1/2*x**2
u_pi_lim = lambda x: 2*x


# m_e = 260

u0 = f_pi**2*m_pi**2

sci = lambda x : "\\times 10^{".join(("%.1e"%x).split("e"))  + "}"


def get_l(m_l):
    u0, ul0, A = get_const_lepton(m_l)

    # x = mu_I / m_pi, y = mu_l / m_l
    eq_l = lambda x, y : A * (y**2 - 1)**(3/2) - 1*x*(1 - 1/x**4)
    f_l = lambda x: sqrt(1 + 1/A**(2/3) * (sqrt(1+x**2) * (1-1/(1+x**2)**2))**(2/3)  )

    u = lambda x: u_pi(sqrt(1+x**2)) + ul0/u0 * ul(sqrt(f_l(x)**2 -1))
    p = lambda x: p_pi(sqrt(1+x**2)) + ul0/u0 * pl(sqrt(f_l(x)**2 -1))

    K1 = 8/15*(2/A)**(5/3)
    K2 = 16/3/A
    pl_lim = lambda x: K1 * x**(5/3)
    ul_lim = lambda x: K2 * x

    return f_l, p, u, pl_lim, ul_lim, A, ul0


N = 400

def plot_mus(m_l=m_e, name="e"):
    x = np.linspace(0, 2, N)
    fig, ax = plt.subplots(figsize=(8, 6))

    f_l, p, u, pl_lim, ul_lim, A, ul0 = get_l(m_l)
    ax.plot(sqrt(1+x**2), f_l(x), "k")
    ax.set_xlabel("$\\mu_I/m_\\pi$")
    ax.set_ylabel("$\mu_"+name+"/m_"+name+"$")

    ax.set_title("$\\quad A =" +sci(A)+"$")
    name = name.replace("\\", "")

    fig.savefig("figurer/charge_neutrality/chemical_potential_"+name+".pdf", bbox_inches="tight")


# limit
def plot_mus_lim(m_l=m_e, name="e"):
    x = np.logspace(-6, 2, N)
    fig, ax = plt.subplots(figsize=(12, 8))

    f_l, p, u, pl_lim, ul_lim, A, ul0 = get_l(m_l)
    ax.plot(sqrt(1+x**2)-1, f_l(x)-1)
    ax.set_xlabel("$\\mu_I/m_\\pi$")
    ax.set_ylabel("$\mu_\\ell/m_\\ell$")

    ax.plot(x**2/2, (2*x**2/A)**(2/3)/2, "r-.")
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.show()


def plot_lepton(name="\\ell"):
    fig, ax = plt.subplots(figsize=(8, 5))

    f_l, p, u, pl_lim, ul_lim, A, ul0 = get_l(100)
    x = np.linspace(0, 0.5, N)
    y = sqrt(f_l(x)**2 - 1)
    ax.plot(pl(y), ul(y), label="$u_{\\ell, \mathrm{nr}}(p)$")

    x = (sqrt(1+ x**2)-1)*2
    ax.plot(pl_lim(x), ul_lim(x), "k--", label="$u_{\\ell, \\mathrm{nr}}$")

    # assert (y[0]==0)
    tit = "$\\frac{\mu_"+name+"}{m_"+name+"}  \in [1,%.2f], \\quad" %(1+x[-1])+"$"
    ax.set_title(tit)
    ax.set_xlabel("$p/u_{\\ell,\mathrm{nr}}$")
    ax.set_ylabel("$u/u_{\\ell,\mathrm{nr}}$")

    ax.legend()
    fig.savefig("figurer/charge_neutrality/eos_lepton.pdf", bbox_inches="tight")



def plt_tot(m_l=m_e, name="e"):
    fig, ax = plt.subplots(figsize=(6, 6))

    f_l, p, u, pl_lim, ul_lim, A, ul0 = get_l(m_l)
    x = np.logspace(-5, 2, N, dtype=np.longdouble())
    ax.plot(p(x), u(x), label="$u(p)$")

    a = ul0/u0
    p = a*pl_lim(x)
    ax.plot(p, u_pi_lim(x) + a*ul_lim(x), "k--", label="$p\propto u^{5/3}$")

    p = p_pi_lim(x)
    ax.plot(p, u_pi_lim(x) + a*ul_lim(x), "r-.", label="$p\propto u^{2}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$p/u_0$")
    ax.set_ylabel("$u/u_0$") 
    ax.legend()

    name = name.replace("\\", "")
    fig.savefig("figurer/charge_neutrality/eos_lepton_limits"+name+".pdf", bbox_inches="tight")

 

def gen_eos_list(xm1, m_l=m_e, name="e"):
    f_l, p, u, pl_lim, ul_lim, A, ul0 = get_l(m_l)

    ulst = u(xm1)
    plst = p(xm1)

    # Can only interpolate with unique points
    assert len(np.unique(plst)) == len(plst)
    assert len(np.unique(ulst)) == len(ulst)

    print(np.sum(np.diff(plst[:100])<0))

    plt.plot(plst, ulst)
    plt.show()
    np.save("pion_star/data/eos_"+name, [xm1, plst, ulst])


def load_eos_list(name):
    x, p, u = np.load("pion_star/data/eos_"+name+".npy")
    return p, u



N = 10000
def plot_all(r, name="nr"):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = 1 + np.logspace(*r, N-1)
    x = np.concatenate([[1], x])

    ue, pe = load_eos_list("e")
    umu, pmu = load_eos_list("mu")

    ax.plot(p_pi(x), u_pi(x), label="$u_\\pi(p)$")
    ax.plot(pe, ue, "k--", label="$u_e(p)+u_\\pi(p)$")
    ax.plot(pmu, umu, "r-.", label="$u_\\mu(p)+u_\\pi(p)$")

    ax.set_xlabel("$p/u_0$")
    ax.set_ylabel("$u/u_0$")

    ax.legend()
    fig.savefig("figurer/charge_neutrality/eos_"+name+".pdf", bbox_inches="tight")


# plot_mus(m_l=m_mu, name="\\mu")
# plot_mus(m_l=m_e, name="e")
# plot_mus_lim(m_l=m_mu, name="\\mu")
# plot_mus_lim(m_l=m_e, name="e")

# plot_lepton()

plt_tot(m_l=m_e, name="e")
plt_tot(m_l=m_mu, name="\\mu")

# plot_all()

# r = (-13, 3)
# x = 1 + np.logspace(*r, N-1)
# gen_eos_list(x, m_e, name="e")


# r = (-12, 2)
# xm1 = np.logspace(*r, N-1, dtype=np.longdouble())
# xm1 = np.concatenate([[0], xm1])
# gen_eos_list(xm1, m_mu, name="mu")

# plot_all(r, name="ur")


