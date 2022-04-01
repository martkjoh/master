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


# x = mu_\ell
xf = lambda x: sqrt(x**2 - 1)
pl = lambda x: 1 / 3 * ((2*xf(x)**3 - 3*xf(x)) * x + 3*arcsinh(xf(x))) 
ul = lambda x: (2*xf(x)**3 + xf(x)) * x - arcsinh(xf(x))

# pl = lambda x: 1 / 3 * ((2*x**3 - 3*x) * sqrt(1 + x**2) + 3*arcsinh(x))
# ul = lambda x: (2*x**3 + x) * sqrt(1 + x**2) - arcsinh(x) 

# mu_\ell = sqrt(1+x) = 1+x/2
pl_lim0 = lambda x: 8/15*x**(5)
ul_lim0 = lambda x: 8/3*x**3

# x = mu_I
p_pi = lambda  x: 1/2 * (x - 1/x)**2
u_pi = lambda x: 1/2 * (2 + x**2 - 3/x**2)

# mu_I = sqrt(1+x) = 1 + x/2
p_pi_lim = lambda x: 1/2*x**2
u_pi_lim = lambda x: 2*x


# m_e = 260

u0 = f_pi**2*m_pi**2

sci = lambda x : "\\times 10^{".join(("%.1e"%x).split("e"))  + "}"


def get_l(m_l):
    u0, ul0, A = get_const_lepton(m_l)

    # x = mu_I / m_pi, y = mu_l / m_l
    eq_l = lambda x, y : A * (y**2 - 1)**(3/2) - 1*x*(1 - 1/x**4)
    # mu_\ell(mu_I)
    f_l = lambda x: sqrt(1 + 1/A**(2/3) * (x * (1-1/x**4))**(2/3))

    # mu_i = sqrt(1+x**2)
    u = lambda x: u_pi(x) + ul0/u0 * ul(f_l(x))
    p = lambda x: p_pi(x) + ul0/u0 * pl(f_l(x))

    K1 = 8/15*(2/A)**(5/3)
    K2 = 16/3/A
    # mu_I = sqrt(1+x) = 1 + x/2
    # in units of u0
    pl_lim = lambda x: K1 * x**(5/3)
    ul_lim = lambda x: K2 * x

    return f_l, p, u, pl_lim, ul_lim, A, ul0


N = 400

def plot_mus(m_l=m_e, name="e"):
    x = 1+np.linspace(0, 2, N)
    fig, ax = plt.subplots(figsize=(8, 6))

    f_l, p, u, pl_lim, ul_lim, A, ul0 = get_l(m_l)
    ax.plot(x, f_l(x), "k")
    ax.set_xlabel("$\\mu_I/m_\\pi$")
    ax.set_ylabel("$\mu_"+name+"/m_"+name+"$")

    ax.set_title("$\\quad A =" +sci(A)+"$")
    name = name.replace("\\", "")

    fig.savefig("figurer/charge_neutrality/chemical_potential_"+name+".pdf", bbox_inches="tight")


# limit
def plot_mus_lim(m_l=m_e, name="e"):
    x = 1+np.logspace(-20, 4, N)
    fig, ax = plt.subplots(figsize=(12, 8))

    f_l, p, u, pl_lim, ul_lim, A, ul0 = get_l(m_l)
    ax.plot(x-1, f_l(x)-1, lw=5, alpha=0.6)
    ax.set_xlabel("$\\mu_I/m_\\pi$")
    ax.set_ylabel("$\mu_\\ell/m_\\ell$")

    eps = (x-1)
    ax.plot(eps/2, (2*eps/A)**(2/3)/2, "k--", lw=2)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.show()


def plot_lepton(name="\\ell"):
    fig, ax = plt.subplots(figsize=(8, 5))

    f_l, p, u, pl_lim, ul_lim, A, ul0 = get_l(100)
    x = 1+np.linspace(0, 0.2, N)
    ax.plot(pl(x), ul(x), label="$u_\\ell(p)$")

    y = sqrt(x**2-1)
    ax.plot(pl_lim0(y), ul_lim0(y), "k--", label="$u_{\\ell, \\mathrm{nr}}(p)$")

    # assert (y[0]==0)
    tit = "$\\frac{\mu_"+name+"}{m_"+name+"}  \in [1,%.2f], \\quad" %(1+x[-1])+"$"
    ax.set_title(tit)
    ax.set_xlabel("$p/u_{\\ell,0}$")
    ax.set_ylabel("$u/u_{\\ell,0}$")

    ax.legend()
    fig.savefig("figurer/charge_neutrality/eos_lepton.pdf", bbox_inches="tight")



def plt_tot(m_l=m_e, name="e"):
    fig, ax = plt.subplots(figsize=(6, 6))

    f_l, p, u, pl_lim, ul_lim, A, ul0 = get_l(m_l)
    x = 1+np.logspace(-11, 2, N, dtype=np.longdouble())
    ax.plot(p(x), u(x), label="$u(p)$")

    a = ul0/u0
    x = (x-1)*2
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

 

def gen_eos_list(r, m_l=m_e, name="e"):
    print(N)
    x = 1 + np.logspace(*r, N-1, dtype=np.longdouble())
    x = np.concatenate([[1], x])

    f_l, p, u, pl_lim, ul_lim, A, ul0 = get_l(m_l)

    ulst = u(x)
    plst = p(x)

    plt.loglog(plst, ulst)
    plt.show()
    print(plst[:100])
    print(np.diff(plst)[:100])

    # Can only interpolate with unique points
    assert len(np.unique(plst)) == len(plst)
    assert len(np.unique(ulst)) == len(ulst)
    assert np.sum(np.diff(plst)<0) == 0
    assert np.sum(np.diff(ulst)<0) == 0
    


    np.save("pion_star/data/eos_"+name, [x, plst, ulst])


def load_eos_list(name):
    x, p, u = np.load("pion_star/data/eos_"+name+".npy")
    return p, u



N = 10000
def plot_all(r, name="nr"):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = 1 + np.logspace(*r, N-1)

    f_l, p, u, pl_lim, ul_lim, A, ul0 = get_l(m_e)
    pe = p(x)
    ue = u(x)
    f_l, p, u, pl_lim, ul_lim, A, ul0 = get_l(m_mu)
    pmu = p(x)
    umu = u(x)

    ax.plot(p_pi(x), u_pi(x), label="$u_\\pi(p)$")
    ax.plot(pe, ue, "k--", label="$u_e(p)+u_\\pi(p)$")
    ax.plot(pmu, umu, "r-.", label="$u_\\mu(p)+u_\\pi(p)$")

    ax.set_xlabel("$p/u_0$")
    ax.set_ylabel("$u/u_0$")

    ax.legend()
    fig.savefig("figurer/charge_neutrality/eos_"+name+".pdf", bbox_inches="tight")




test((8, 8.5))
# test2()

# plot_mus(m_l=m_mu, name="\\mu")
# plot_mus(m_l=m_e, name="e")
# plot_mus_lim(m_l=m_mu, name="\\mu")
# plot_mus_lim(m_l=m_e, name="e")

plot_lepton()
 
# plt_tot(m_l=m_e, name="e")
# plt_tot(m_l=m_mu, name="\\mu")

N = 1000
# r = (-13, 3)
# gen_eos_list(r, m_e, name="e")
r = (-10, 5)
# gen_eos_list(r, m_mu, name="mu")

# r = (-5,-2)
# plot_all(r, name="nr")
# r = (-2, 1)
# plot_all(r, name="ur")


