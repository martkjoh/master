import numpy as np
import sympy as sp
from sympy import lambdify, pi, sin, cos, sqrt, log as ln
from scipy.integrate import quad, quadrature
from scipy.optimize import newton
from scipy.interpolate import splev, splrep

from matplotlib import pyplot as plt
from tqdm import tqdm
import sys

if __name__=="__main__":
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
def gen_alpha(N=1000, r=(-8, np.log10(5.5))):
    ### Free energy diff alpha

    F1_diff_a = F1.diff(a)
    dF_fin_diff_a = dF_fin.diff(a)
    F_diff_a = get_total_nlo(F1_diff_a, dF_fin_diff_a)

    x0 = np.linspace(0, 1, 100)
    x = 1+np.logspace(*r, N-100) # change 1 to point of phase transition?
    x = np.concatenate([x0, x])

    mus = x * m_pi

    alphas = np.empty_like(mus)
    for i, mu in enumerate(tqdm(mus)):
        a0 = alpha_0(mu/m_pi)[0]
        a1, r = newton(lambda ai: F_diff_a(mu, ai), a0, full_output=True)
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
    nI = np.array([-F_diff_muI(mu, a)*m_pi for mu, a in zip(mus, alphas)])
    np.save("pion_star/data/nlo_nI", nI)


def gen_p():
    mus, alphas = np.load("pion_star/data/nlo_mu_alpha.npy")
    F = get_total_nlo(F1, dF_fin)
    P = np.array([-F(mu, a) + F(0, 0) for mu, a in zip(mus, alphas)])
    np.save("pion_star/data/nlo_p", P)



def gen_u():
    mus, alphas = np.load("pion_star/data/nlo_mu_alpha.npy")
    nI = np.load("pion_star/data/nlo_nI.npy")
    P = np.load("pion_star/data/nlo_p.npy")
    u = -P + mus/m_pi*nI

    np.save("pion_star/data/nlo_u", u)


def gen_eos():
    N = 1000
    mus, alphas = np.load("pion_star/data/nlo_mu_alpha.npy")
    x = mus/m_pi
    P = np.load("pion_star/data/nlo_p.npy")
    u = np.load("pion_star/data/nlo_u.npy")
    
    # The transition do not happen exactly at m_pi, it only does so to
    # NLO. NNLO terms may contribute numrically. These filte out terms 
    # before in the vacuum phase
    mask = (P>1e-10)
    i = np.where(np.logical_not(mask))[0][-1]
    mask[i] = True # Add one (0, 0) point
    P = P[mask]
    u = u[mask]
    x = x[mask]
    
    # Both p and u should be strictly increasing
    assert np.sum(np.diff(P)>0) == len(P)-1
    assert np.sum(np.diff(u)>0) == len(u)-1

    np.save("pion_star/data/eos_nlo", [x, P, u])




if __name__=="__main__":
    # gen_alpha()
    gen_nI()
    gen_p()
    gen_u()
    gen_eos() 