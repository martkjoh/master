import numpy as np
import sympy as sp
from sympy import lambdify, pi, sin, cos, sqrt, log as ln
from scipy.integrate import quad, quadrature
from scipy.optimize import newton
from scipy.interpolate import splev, splrep

from matplotlib import pyplot as plt
from tqdm import tqdm
import sys

sys.path.append(sys.path[0] + "/..")
from integrate_tov import get_u
from constants import m_pi, m_rho, Lr as Lr_num
from nlo_const import get_nlo_const
from spectrum import * 

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)

# m_pi = 131
# m_K = 481
# f_pi = 128/np.pi
# m_S = sqrt(2*m_K**2 - m_pi**2)

u0 = (f_pi*m_pi)**2

def lo(x):
    # substitute everything except a, muI
    c = [m, dm, mS, f, mrho, muS]
    c_num = [m_pi, 0, m_S, f_pi, m_rho, 0]
    for i, c0 in enumerate(c_num):
        x = x.subs(c[i], c0)
    for i, L in enumerate(Lr_num):
        x = x.subs(Lr[i], L)
    return x

def nlo(x):
    # substitute everything except a, muI
    m_nlo, mS_nlo, f_nlo = get_nlo_const()
    # mS_nlo = sqrt(2*m_K**2- m_nlo**2)
    c = [m, dm, mS, f, mrho, muS]
    c_num = [m_nlo, 0, mS_nlo, f_nlo, m_rho, 0]
    for i, c0 in enumerate(c_num):
        x = x.subs(c[i], c0)
    for i, L in enumerate(Lr_num):
        x = x.subs(Lr[i], L)
    return x

def l(x):
    return lambdify((muI, a), nlo(x))

def lp(x):
    return lambdify((p, muI, a), nlo(x))

integral = lambda f: (
    lambda *args: quad(f, 0, np.inf, args=args, epsabs=1e-5, epsrel=1e-5)[0]
    )


def f_from_df(dF):
    dF_num = lp(dF)
    dfC = lambda p, m, a: dF_num(p+0j, m+0j, a+0j)
    def dF(p, m, a):
        d = dfC(p, m, a)
        a, b = d.real, d.imag
        if np.abs(d)>1e-15: Exception("large im : %.3e + %.3e i" % (a, b))
        return a
    return integral(dF)


#### Free energy
#### LO

F_0_2 = - f**2/2 * (2*m**2*cos(a) + muI**2*sin(a)**2 + mS**2) / u0


#### NLO
mtilde1_sq = m1_sq + 1/4 * m12**2
mtilde2_sq = m2_sq + 1/4 * m12**2
mtilde45_sq = m4_sq + 1/4 * m45**2
mtilde67_sq = m6_sq + 1/4 * m67**2


E1_sq = p**2 + mtilde1_sq
E2_sq = p**2 + mtilde2_sq

Lr = [sp.symbols("H_2")]
[Lr.append(sp.symbols("L_%d"%i)) for i in range(1,9)]


F_0_4 = -(
    +2*(2*Lr[1] + 2*Lr[2] + Lr[3]) * (muI**2*sin(a)**2)**2 
    +4*Lr[4] * (2*m**2*cos(a) + mS**2) * (muI**2 * sin(a)**2)
    +4*Lr[5] * (m**2*cos(a)) * (muI**2 * sin(a)**2)
    +4*Lr[6] * (2*m**2*cos(a) + mS**2)**2
    +2*Lr[8] * (2*m**4*cos(2*a) + 2*dm**4 + mS**4)
    +1*Lr[0] * (2*m**4 + 2*dm**4 + mS**4)
) / u0

F_ln = \
    -1/(4*pi)**2 * (
        + 1/2 * ( 1/2 + ln(mrho**2 / m3_sq) ) * m3_sq**2
        + 1/4 * ( 1/2 + ln(mrho**2 / m8_sq) ) * m8_sq**2
        + 1/4 * ( 1/2 + ln(mrho**2 / mtilde1_sq) ) * mtilde1_sq**2
        + 1/2 * ( 1/2 + ln(mrho**2 / mtilde45_sq) ) * mtilde45_sq**2
        + 1/2 * ( 1/2 + ln(mrho**2 / mtilde67_sq) ) * mtilde67_sq**2
) / u0


dF_fin = 1 / (2 * pi)**2 * p**2 * (
    sqrt(Epip_sq) + sqrt(Epim_sq) - sqrt(E1_sq) - sqrt(E2_sq)
    ) / u0


F1 = (
    F_0_2
    + F_0_4
    + F_ln
)

def get_total_nlo(F1, dF_fin):
    F_fin = f_from_df(dF_fin)
    g = l(F1)

    return lambda muI, a: g(muI, a) + F_fin(muI, a)



##############
# find alpha #
##############
def gen_alpha(N=1000, r=(0, 5)):
    ### Free energy diff alpha

    F1_diff_a = F1.diff(a)
    dF_fin_diff_a = dF_fin.diff(a)
    F_diff_a = get_total_nlo(F1_diff_a, dF_fin_diff_a)

    mus = np.linspace(*r, N) * m_pi
    alphas = np.empty_like(mus)
    for i, mu in enumerate(tqdm(mus)):
        a0 = alpha_0(mu/m_pi)[0]
        F_diff_a_of_a = lambda ai: F_diff_a(mu, ai)
        a1, r = newton(F_diff_a_of_a, a0, full_output=True)
        print((a0, a1, mu/(4*np.pi*f_pi)))
        print(r)
        alphas[i] = a1

    alphas = np.array(alphas)

    np.save("pion_star/data/nlo_mu_alpha", [mus, alphas])
    return mus, alphas


def get_alpha_nlo():
    """ Create alpha_nlo(muI) with splines """
    mus, alphas = np.load("pion_star/data/nlo_mu_alpha.npy")
    mus = mus/m_pi
    mask = (alphas != 0)
    tck = splrep(mus[mask], alphas[mask], s=0, k=1)
    mu_min = np.min(mus[mask])

    def alpha(mu):
        """mu = mu / m_pi"""
        assert np.all(mu<=mus[-1]),"a-value outside interpolation area, a=%.3e"%mus[-1]
        t = type(mu)
        mu = np.atleast_1d(mu)
        mask = mu>mu_min
        alphas = np.zeros_like(mu)
        if not len(alphas[mask])==0:
            alphas[mask] = splev(mu[mask], tck)
        if t!=np.ndarray: alphas=t(alphas)
        return  alphas

    return alpha



def gen_nI():
    # Free energy diff mu_I
    F1_diff_muI = F1.diff(muI)
    dF_fin_diff_muI = dF_fin.diff(muI)
    F_diff_muI = get_total_nlo(F1_diff_muI, dF_fin_diff_muI)

    mus, alphas = np.load("pion_star/data/nlo_mu_alpha.npy")
    nuIs = np.array([-F_diff_muI(mu, a) for mu, a in zip(mus, alphas)])
    np.save("pion_star/data/nlo_nI", nuIs)

    plt.plot(mus/m_pi, nuIs*m_pi)
    def n(mu):
        n=np.zeros_like(mu)
        mask = mu>1
        n[mask]=(mu[mask]**2 - 1/mu[mask]**2)/mu[mask]
        return n
    plt.plot(mus/m_pi, n(mus/m_pi))
    plt.show()


def gen_F():
    mus, alphas = np.load("pion_star/data/nlo_mu_alpha.npy")
    
        
    def F0(mu):
        a0 = alpha_0(mu/m_pi)
        return -f_pi**2/2*(mu**2*np.sin(a0)**2 + 2*m_pi**2*np.cos(a0) + m_S**2)
        
    plt.plot(mus/m_pi, F0(mus)/u0, label="F0")


    F_0_2_num = l(F_0_2)
    F_0_4_num = l(F_0_4)
    F_ln_num = l(F_ln)
    F_fin_num = f_from_df(dF_fin)
    Fs = [
        F_0_2_num, 
        F_0_4_num, 
        F_ln_num,
        F_fin_num
        ]


    Fs_array = [[F(mu, a) for mu, a in zip(mus, alphas)] for F in Fs]
    labels = [
        "F02", 
        "F04", 
        "Fln",
        "Ffin"]
    [plt.plot(mus/m_pi, F_array, label= labels[i]) for i, F_array in enumerate(Fs_array)]


    F1 = np.sum(np.array(Fs_array), axis=0)
    plt.plot(mus/m_pi, F1, label="F1")

    plt.legend()
    plt.show()


def gen_u():
    mus, alphas = np.load("pion_star/data/nlo_mu_alpha.npy")
    nI = np.load("pion_star/data/nlo_nI.npy")
    p = np.load("pion_star/data/nlo_p.npy")
    
    u = -p + mus*nI

    np.save("pion_star/data/nlo_u", u)

    plt.plot(p, u, "k--")
    plt.show()


def gen_eos(r=(-10, 0.7)):
    N = 1000

    alpha = get_alpha_nlo()

    # x = 1+np.logspace(*r, N-1)
    # x = np.concatenate([[1.,], x])
    # i = np.where(alpha(x)>1e-10)[0][0]
    # x_min = x[i]

    x_min = 1

    x = x_min+np.logspace(*r, N-1)
    x = np.concatenate([[x_min,], x])
    a = alpha(x)
    mu = x*m_pi

    F1_diff_muI = F1.diff(muI)
    dF_fin_diff_muI = dF_fin.diff(muI)
    F_diff_muI = get_total_nlo(F1_diff_muI, dF_fin_diff_muI)
    nI = np.array([-F_diff_muI(x, y) for x, y in zip(mu, a)])
    
    F = get_total_nlo(F1, dF_fin)
    p = np.array([-F(x, y) + F(0, 0) for x, y in zip(mu, a)])
    u = -p+mu*nI

    mask = (p>1e-10)
    i = np.where(np.logical_not(mask))[0][-1]
    mask[i] = True # Add one (0, 0) point
    p = p[mask]
    u = u[mask]
    x = x[mask]
    print(p)
    print(i)

    assert np.sum(np.diff(p)>0) == len(p)-1
    assert np.sum(np.diff(u)>0) == len(u)-1

    np.save("pion_star/data/eos_nlo", [x, p, u])
    return p, u



if __name__=="__main__":
    print(4*np.pi*f_pi/m_pi)
    mus, alpha = gen_alpha(N=100, r=(1, 8))
    # mus, alpha = gen_alpha(N=1000, r=(.9, 1.1))
    # x = mus/m_pi
    x = np.linspace(.9, 1.1, 10000)
    alpha = get_alpha_nlo()
    plt.plot(x, alpha_0(x), "k--")
    plt.plot(x, alpha(x))
    plt.show()

    ##

    # gen_alpha(N = 1000, r=(0, 7))
    # gen_eos() 

    # _, p, u = np.load("pion_star/data/eos_nlo.npy")
    # p = np.linspace(0, 1e-4, 10000)
    # u = get_u("pion_star/data/eos_nlo.npy")
    # plt.plot(p, u(p))
    # plt.show()

    # save_eos(r=(-6, 0.5))

    # _, P, u2 = np.load("pion_star/data/eos_nlo.npy")
    # print(P)
    # plt.plot(P, u2)

    # u1 = get_u("pion_star/data/eos_nlo.npy")
    # plt.plot(P, u1(P), "k--")

    # plt.show()
