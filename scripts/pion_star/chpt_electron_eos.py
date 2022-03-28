import numpy as np
from numpy import pi, sqrt, arcsinh
from scipy.optimize import newton
from matplotlib import pyplot as plt
import sys

sys.path.append(sys.path[0] + "/..")
from constants import get_const_lepton, m_e, m_pi
from integrate_tov import get_u

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)


xf = lambda x: sqrt(x**2 - 1)
ul = lambda x: (2*xf(x)**3 + xf(x)) * sqrt(1 + xf(x)**2) - arcsinh(xf(x)) 
pl = lambda x: 1 / 3 * ((2*xf(x)**3 - 3*xf(x)) * sqrt(1 + xf(x)**2) + 3*arcsinh(xf(x)))

p_pi = lambda x: 1/2 * (x - 1/x)**2
u_pi = lambda x: 1/2 * (2 + x**2 - 3/x**2)

m_e = 0.5
m_e = 2000
# m_e = 260

u0, ul0, A = get_const_lepton(m_e)

# x = mu_I / m_pi, y = mu_l / m_l
eq_e = lambda x, y : A * (y**2 - 1)**(3/2) - 1*x*(1 - 1/x**4)
f_e = lambda x: sqrt( 1 + 1/A**(2/3) * (x * (1-1/x**4))**(2/3)  )


u_e = lambda x: u_pi(x) + ul0/u0 * ul(f_e(x))
p_e = lambda x: p_pi(x) + ul0/u0 * pl(f_e(x))

N = 100

m = np.log(10)


N = 1000
x = np.linspace(0, 1, N)[1::]



p_pi_lim = lambda x: 1/2*x**2
u_pi_lim = lambda x: 2*x

K1 = 8/15*(2/A)**(5/3)
K2 = 16/3/A
pl_lim = lambda x: K1 * x**(5/3) 
ul_lim = lambda x: K2 * x


N = 400

# limit
def plot_mus():
    x = 1+np.logspace(-10, 5, N)
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(x-1, f_e(x)-1)
    ax.set_xlabel("$\\mu_I/m_\\pi$")
    ax.set_ylabel("$\mu_\\ell/m_\\ell$")

    eps = (x-1)*2
    ax.plot(eps/2, (2*eps/A)**(2/3)/2, "r-.")
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.show()


def plot_lepton():
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.linspace(0, A/10, N)
    y_pi = 1+x/2
    y = f_e(y_pi)
    y = 1 + (2/A*x)**(2/3)/2
    ax.plot(pl(y), ul(y))
    ax.plot(pl_lim(x), ul_lim(x), "k--")
    tit = "$\\frac{\mu_\\ell}{m_\\ell}-1  \in [%.2f," %(y[0]-1) +"%.2f], \\quad" %(y[-1]-1)\
        +  "\\frac{\mu_I}{m_\\pi}-1 \in [" + "\\times 10^{".join(("%.1e"%(y_pi[0]-1)).split("e"))  + "},"\
        + "\\times 10^{".join(("%.1e"%(y_pi[-1]-1)).split("e")) + "}]$"
    ax.set_title(tit)

    print(y_pi)


    plt.show() 


def plot_pion():
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.linspace(0, .35, N)
    y = 1+x/2
    ax.plot(p_pi(y), u_pi(y))
    ax.plot(p_pi_lim(x), u_pi_lim(x), "k--")

    plt.show() 


def plt_tot():
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.logspace(-10, 2, N, dtype=np.longdouble())
    y = 1+x/2
    y_l = f_e(y)
    ax.plot(p_e(y), u_e(y))

    a = ul0/u0
    p = a*pl_lim(x)
    ax.plot(p, u_pi_lim(x) + a*ul_lim(x), "k--")

    p = p_pi_lim(x)
    ax.plot(p, u_pi_lim(x) + a*ul_lim(x), "r-.")

    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.show()

 

def gen_eos_list_e(x): 
    ulst = u_e(x)
    plst = p_e(x)

    # Can only interpolate with unique points
    assert len(np.unique(plst)) == len(plst)
    assert len(np.unique(ulst)) == len(ulst)
    np.save("pion_star/data/eos_e", [x, plst, ulst])

    return ulst, plst




N = 1000
def plot_all():
    fig, ax = plt.subplots(figsize=(12, 8))
    x = 1 + np.logspace(-7, 1, N-1)
    x = np.concatenate([[1], x])
    u, p = gen_eos_list_e(x)

    ax.plot(p, u)

    ax.plot(p_pi(x), u_pi(x), "k--")
    plt.show()


# plot_mus()
# plot_lepton()
# plot_pion()
# plt_tot()

plot_all()