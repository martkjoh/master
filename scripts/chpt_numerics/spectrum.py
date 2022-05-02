import numpy as np
import sympy as sp
from sympy import lambdify, pi, sin, cos, sqrt, log as ln
import sys

sys.path.append(sys.path[0] + "/..")
from constants import f_pi, m_S, m_pi, Dm, m_Kpm, Dm_EM


# Everything is done in units of m_pi
m, mrho, mS, f = sp.symbols("m, m_rho, m_S, f")
# This is usefull to write rational numbers
one = sp.S.One
a = sp.symbols("alpha")
muI = sp.symbols("mu_I")
muS = sp.symbols("mu_S")
dm = sp.symbols("Delta_m")
p = sp.symbols("p", real=True)

# Insert leading order values for constants, in units of u_0
lo = lambda x: x.subs(m, 1.).subs(f, 1.).subs(mS, m_S/m_pi).subs(dm, Dm/m_pi)
num_lo = lambda x: lambdify((muI, a), lo(x), "numpy")

# Mass-parameters
m1_sq = m**2 * cos(a) - muI**2 * cos(2 * a)
m2_sq = m**2 * cos(a) - muI**2 * cos(a)**2
m3_sq = m**2 * cos(a) + muI**2 * sin(a)**2
m8_sq = one/3*(m**2*cos(a) + 2*mS**2)

m_p_sq = one/2 * (m**2*cos(a) + mS**2 + dm**2)
m_m_sq = one/2 * (m**2*cos(a) + mS**2 - dm**2)
m_mu_p_sq = one/4 * muI**2*cos(2*a) + muI*muS*cos(a) + muS**2
m_mu_m_sq = one/4 * muI**2*cos(2*a) - muI*muS*cos(a) + muS**2

m4_sq = m_m_sq - m_mu_p_sq
m6_sq = m_p_sq - m_mu_m_sq

m12 = 2 * muI * cos(a)
m45 = muI*cos(a) + 2*muS
m67 = muI*cos(a) - 2*muS

msq = [m1_sq, m2_sq, m3_sq, m4_sq, m6_sq, m8_sq]
mij = [m12, m45, m67]

E0_sq = p**2 + m3_sq
Eeta_sq = p**2 + m8_sq


def get_E(m1_sq, m2_sq, m12, p=p):
    M_sq = (m1_sq + m2_sq + m12**2)
    Ep_sq = \
        p**2 \
        + one/2 * M_sq \
        + one/2 * sp.sqrt(4 * p**2 * m12**2 + M_sq**2 - 4*m1_sq*m2_sq)
    Em_sq = \
        p**2 + one/2 * M_sq\
        - one/2 * sp.sqrt(4 * p**2 * m12**2 + M_sq**2 - 4*m1_sq*m2_sq)
    return Ep_sq, Em_sq


Epim_sq, Epip_sq = get_E(m1_sq, m2_sq, m12)
EKm_sq, EKp_sq = get_E(m4_sq, m4_sq, m45)
EK0bar_sq, EK0_sq = get_E(m6_sq, m6_sq, m67)


# first approx to alpha as a function of mu_I, analytical result
def alpha_0(mu):
    mu = np.atleast_1d(mu).astype(float)
    morethan_m = mu**2 > np.ones_like(mu)
    a = np.zeros_like(mu)
    a[morethan_m] = np.arccos((1/mu[morethan_m]**2))
    return a



#### EM version

D = Dm_EM**2/m_pi**2

m1_sq_EM = m**2 * cos(a) - ( muI**2 - D)* cos(2 * a)
m2_sq_EM = m**2 * cos(a) - ( muI**2 - D)* cos(a)**2
m3_sq_EM = m**2 * cos(a) + ( muI**2 - D)* sin(a)**2

m_mu_p_sq_EM = one/4 * muI**2*cos(2*a) + muI*muS*cos(a) + muS**2 +(cos(a)**2 + cos(a))*D/2
m_mu_m_sq_EM = one/4 * muI**2*cos(2*a) - muI*muS*cos(a) + muS**2 -(cos(a)**2 - cos(a))*D/2
m4_sq_EM = m_m_sq - m_mu_p_sq_EM
m6_sq_EM = m_p_sq - m_mu_m_sq_EM


E0_sq_EM = p**2 + m3_sq_EM

Epim_sq_EM, Epip_sq_EM = get_E(m1_sq_EM, m2_sq_EM, m12)
EKm_sq_EM, EKp_sq_EM = get_E(m4_sq_EM, m4_sq_EM, m45)
EK0bar_sq_EM, EK0_sq_EM = get_E(m6_sq_EM, m6_sq_EM, m67)

def alpha_EM(mu):
    mu = np.atleast_1d(mu).astype(float)
    morethan_m = mu**2 > (1+D)*np.ones_like(mu)
    a = np.zeros_like(mu)
    x = 1/ (mu[morethan_m]**2 - D)
    a[morethan_m] = arccos(x)
    return a


#### Free energy
#### LO

F_0_2 = - one/2* f**2 * (muI**2 * sin(a)**2 + 2*m**2 * cos(a) + mS**2)


#### NLO
mtilde1_sq = m1_sq + one/4 * m12**2
mtilde2_sq = m2_sq + one/4 * m12**2
mtilde45_sq = m4_sq + one/4 * m45**2
mtilde67_sq = m6_sq + one/4 * m67**2

E1_sq = p**2 + mtilde1_sq
E2_sq = p**2 + mtilde2_sq

Ls = [sp.symbols("H_2")]
[Ls.append(sp.symbols("L_%d"%i)) for i in range(1,9)]


F_0_4 = \
    2*(2*Ls[1] + 2*Ls[2] + Ls[3]) * muI**4*sin(a)**4 \
    -4*Ls[4]*(2*m**2*cos(a) + mS**2)*muI**2*sin(a)**2 \
    -4*Ls[5]*m**2*muI**2*cos(a)*sin(a)**2 \
    -4*Ls[6]*(2*m + mS)**2 \
    -2*Ls[8]**(2*m**4*cos(2*a) + 2*dm**4 + mS**4) \
    -Ls[0]*(2*m**4 + 2*dm**4 + mS**4)

F_ln = \
    - one/2/(4*pi)**2 *\
    (
        (1 + ln(mrho**2/m3_sq)) * m3_sq**2
        + one/18 * (1 + ln(mrho**2/m8_sq)) * m8_sq**2
        + one/2 * (1 + ln(mrho**2/m1_sq)) * m1_sq**2
        + one/4 * (1 + ln(mrho**2/mtilde45_sq)) * mtilde45_sq**2
        + one/4 * (1 + ln(mrho**2/mtilde67_sq)) * mtilde67_sq**2
)

dF_fin = \
    one/2 * (4 * pi)/(2*pi)**3 * p**2 * (
        sqrt(Epip_sq) + sqrt(Epim_sq) - sqrt(E1_sq) - sqrt(E2_sq)
    )



# Much slower method, more accurate
integral = lambda f: (
    lambda *args: quad(f, 0, 50, args=args, epsabs=1e-5, epsrel=1e-5)[0]
    )

def f_from_df_symb(df_symb):
    df = lambdify((p, mu, a), df_symb, "numpy")

    df_C = lambda p, m, a: df(p+0j, m+0j, a+0j)
    df_R = lambda p, m, a: df_C(p, m, a).real
    
    f = integral(df_R)
    f = np.vectorize(f)
    return f


