import numpy as np
import sympy as sp
from sympy import lambdify, pi, sin, cos, sqrt, log as ln
from scipy.integrate import quad
from scipy.optimize import newton
from matplotlib import pyplot as plt
import sys

sys.path.append(sys.path[0] + "/..")
from constants import m_pi, m_rho, Lr as Lr_num
from nlo_const import get_nlo_const
from spectrum import *


def nlo(x):
    # substitute everything except a, muI
    m_nlo, mS_nlo, f_nlo = get_nlo_const()
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
    lambda *args: quad(f, 0, 1, args=args, epsabs=1e-5, epsrel=1e-5)[0]
    )

def f_from_df(dF):
    dF_num = lp(dF)

    df_C = lambda p, m, a: dF_num(p+0j, m+0j, a+0j)
    df_R = lambda p, m, a: df_C(p, m, a).real
    
    f = integral(df_R)
    f = np.vectorize(f)
    return f


#### Free energy
#### LO

F_0_2 = - one/2* f**2 * (muI**2 * sin(a)**2 + 2*m**2 * cos(a) + mS**2)


#### NLO
mtilde1_sq = m1_sq + one/4 * m12**2
mtilde2_sq = m2_sq + one/4 * m12**2
mtilde45_sq = m4_sq + 1/4 * m45**2
mtilde67_sq = m6_sq + 1/4 * m67**2
# mtilde45_sq =(m**2*cos(a) + mS**2)/2 + 1/4*muI**2*sin(a)**2
# mtilde67_sq = mtilde45_sq

E1_sq = p**2 + mtilde1_sq
E2_sq = p**2 + mtilde2_sq

Lr = [sp.symbols("H_2")]
[Lr.append(sp.symbols("L_%d"%i)) for i in range(1,9)]


F_0_4 = \
    -2*(2*Lr[1] + 2*Lr[2] + Lr[3]) * (muI**2*sin(a)**2)**2 \
    -4*Lr[4] * (2*m**2*cos(a) + mS**2) * (muI**2 * sin(a)**2) \
    -4*Lr[5] * (m**2*cos(a)) * (muI**2 * sin(a)**2)\
    -4*Lr[6] * (2*m**2 + mS**2)**2 \
    -2*Lr[8] * (2*m**4*cos(2*a) + 2*dm**4 + mS**4) \
    -  Lr[0] * (2*m**4 + 2*dm**4 + mS**4)

F_ln = \
    -1/ (2*(4*pi)**2) * \
    (
        +       (1/2 + ln(mrho**2/m3_sq)) * m3_sq**2
        + 1/2 * (1/2 + ln(mrho**2/m8_sq)) * m8_sq**2
        + 1/2 * (1/2 + ln(mrho**2/m1_sq)) * m1_sq**2
        +       (1/2 + ln(mrho**2/mtilde45_sq)) * mtilde45_sq**2
        +       (1/2 + ln(mrho**2/mtilde67_sq)) * mtilde67_sq**2
)

dF_fin = \
    one/(2*pi)**2 * p**2 * (
        sqrt(Epip_sq) + sqrt(Epim_sq) - sqrt(E1_sq) - sqrt(E2_sq)
    )

F_fin = f_from_df(dF_fin)
F1 = \
    F_0_2 \
    + F_0_4 \
    + F_ln

u0 = f_pi**2*m_pi**2
g = l(F1)
F = lambda muI, a: (g(mu, a) + F_fin(muI, a)) / u0

dF1 = F1.diff(a)
ddF_fin = dF_fin.diff(a)
dF_fin = f_from_df(ddF_fin) # now diff wrt a
dg = l(dF1) 
dF = lambda muI, a:  (
    dg(muI, a) 
    + dF_fin(muI, a)
    ) / u0



# mus = np.linspace(0, 2)*m_pi
# alphas = alpha_0(mus/m_pi)
# k = dF(mus, alphas) 
# plt.plot(mus/m_pi, k)
# k = dF(mus, alphas*1.2)
# plt.plot(mus/m_pi, k)
# k = dF(mus, alphas*1.3)
# plt.plot(mus/m_pi, k)
# k = dF(mus, np.ones_like(mus))
# plt.plot(mus/m_pi, k)

# plt.show()


##############
# find alpha #
##############
def find_alpha():
    mus = np.linspace(0.9, 1.1, 200) * m_pi
    alphas = []
    for mu in mus:
        # a0 = alpha_0(mu/m_pi)[0]*1.1
        a0 = 1
        dFmu = lambda a0: dF(mu, a0)
        a1, r = newton(dFmu, a0, full_output=True)
        print((mu/m_pi, a1))
        alphas.append(a1)

    plt.plot(mus/m_pi, alphas)
    plt.plot(mus/m_pi, alpha_0(mus/m_pi), "k--")
    plt.show()


find_alpha()
