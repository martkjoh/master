load("../power_expansion_chpt.sage") 
import numpy as np

# Gell-Mann matrices
l1 = matrix([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]])
l2 = matrix([
        [0, -i, 0],
        [i, 0, 0],
        [0, 0, 0]])
l3 = matrix([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 0]])
l4 = matrix([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]])
l5 = matrix([
        [0, 0, -i],
        [0, 0, 0],
        [i, 0, 0]])
l6 = matrix([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0]])
l7 = matrix([
        [0, 0, 0],
        [0, 0, -i],
        [0, i, 0]])
l8 = matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -2]]) / sqrt(3)


one = matrix.identity(l1.dimensions()[0])

l = [l1, l2, l3, l4, l5, l6, l7, l8]

# Isospin chemical potential current

var("e", latex_name="\\varepsilon", domain="real")
var("d", latex_name="\\delta", domain="real")
var("a", latex_name="\\alpha", domain="real")

muS = var("muS", latex_name="\\mu_S")
muB = var("muB", latex_name="\\mu_B")
muI = var("muI", latex_name="\\mu_I")

mu = diagonal_matrix([muB/3 + muI/2, muB/3 - muI/2, muB/3 - muS])

v_mu = d*mu

# e A_mu
var("qe", latex_name="e", domain="real")
var("A_mu", latex_name="\\mathcal A_\\mu", domain="real")
var("C")
Q = diagonal_matrix( (2/3, -1/3, -1/3))

v_em = qe*A_mu*Q

# pi_a tau_a

x = var("x")
var("f")


p = vector([
	function("phi"+str(i), latex_name="\\varphi_"+str(i))(x) 
	for i  in range(1, len(l)+1)
])
p_s = e * sum([l[i]*p[i] for i in range(len(l))])

pvar = vector([
	var(
		"phi"+str(i), 
		latex_name="\\varphi_"+str(i),
		domain="real"
	) for i  in range(1, len(l)+1)
])
dpvar = vector([
	var(
		"dphi"+str(i), 
		latex_name="\\partial_\\mu\\varphi_"+str(i), 
		domain="real"
	) for i  in range(1, len(l)+1)
])



var("dm", latex_name="\\Delta m")
var("mbar", latex_name="\\bar m")
var("mS", latex_name="m_S")
chi = diagonal_matrix((mbar^2 - dm^2, mbar^2 + dm^2, mS^2))

exp_l_a = lambda l, a: (one - l**2) + l**2 * cos(a) + I * l * sin(a)

A = lambda l: exp_l_a(l, a/2)
U = lambda n: EXP(I * p_s, n)
U_d = lambda n: EXP(-I * p_s, n)

SIGMA = lambda n, A: mat_prep(A * U(n) * A, n)
SIGMA_d = lambda n, A: mat_prep(A.C.T * U_d(n) * A.C.T, n)


def sub(S, p, pvar):
    for i, p0 in enumerate(pvar):
        for j in range(3):
            for k in range(3):
                S[j, k]=S[j, k].subs(p[i]==p0)


def get_S(n, li, indx="mu"):
    if li==None: Ai = one
    else: Ai = A(li)
    S = SIGMA(n, Ai)
    Sd = SIGMA_d(n, Ai)
    dS = diff(S, x)
    dSd = diff(Sd, x)

    sub(S, p, pvar)
    sub(Sd, p, pvar)

    sub(dS, diff(p, x), dpvar)
    sub(dSd, diff(p, x), dpvar)

    return S, Sd, dS, dSd


