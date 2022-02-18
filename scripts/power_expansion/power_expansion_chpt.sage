
# Pauli matrices

import numpy as np
s1 = matrix(
    [
        [0, 1],
        [1, 0]
    ]
)
s2 = matrix(
    [
        [0, -i],
        [i, 0]
    ]
)
s3 = matrix(
    [
        [1, 0],
        [0, -1]
    ]
)

one = matrix.identity(s1.dimensions()[0])

s = [s1, s2, s3]

# Hack
# Needed to take conjugate of pi functions
# Do not know of a better way

def Id(self, x): return self(x)

p = vector([
    function("pi"+str(i), latex_name="\\pi_"+str(i), conjugate_func=Id)(x) for i  in range(1, 4)
])


# Isospin chemical potential current

var("e", latex_name="\\varepsilon", domain="real")
var("d", latex_name="\\delta", domain="real")
var("mu", latex_name="\\mu_I", domain="real")
var("a", latex_name="\\alpha", domain="real")

v_I = 1/2*d*mu*s3 


# Charge matrix
Q = 1/6 * one + 1/2*s3

# e A_mu
var("eA", latex_name="e \\mathcal A_\\mu", domain="real")
v_EM = eA * Q


# chi with mass

var("dm", latex_name="\\Delta m", domain="real")
var("mm", latex_name="\\bar m", domain="real")
var("B0", latex_name="B_0", domain="real")

chi = (mm^2 * one + dm^2 * s3)


# pi_a tau_a

pi_s = e * sum([s[i]*p[i] for i in range(len(s))])


POW = lambda A, n : matrix.identity(A.dimensions()[0]) if (n == 0) else A * POW(A, n-1)
EXP = lambda A, n : sum([POW(A, i)/factorial(i) for i in range(n+1)])

A_a = one*cos(a/2) + I * s1 * sin(a/2)
U = lambda n: EXP(I * pi_s/2, n)
SIGMA = lambda n: mat_prep(A_a * U(n) * U(n) * A_a, n)


acom = lambda A1, A2 : A1 * A2 + A2 * A1
com = lambda A1, A2 : A1 * A2 - A2 * A1


# Project matrices onto basis of pauli matrices
def proj(A):
    A1 = A.trace()/2
    A = A - one * A1
    v = [A1, acom(A, s1)[0, 0]/2, acom(A, s2)[0, 0]/2, acom(A, s3)[0, 0]/2]
    v = [a.full_simplify().trig_reduce() for a in v]
    return v 


# Helper functions

def mat_series(mat, x, n):
    d = mat.dimensions()
    for i in range(0, d[0]):
        for j in range(0, d[1]):
            mat[i, j] = mat[i, j].series(x, n).truncate()

def mat_simp(mat):
    d = mat.dimensions()
    for i in range(0, d[0]):
        for j in range(0, d[1]):
            mat[i, j] = mat[i, j].full_simplify()
            mat[i, j] = mat[i, j].trig_reduce()

def mat_prep(mat, n):
    mat_series(mat, e, n)
    mat_simp(mat)
    return mat

def print_e(elem):
    coeff = elem.coefficients(e)
    for i in range(len(coeff)):
        print(e^coeff[i][1], ":")
        c = coeff[i][0].coefficients(mu)
        for k in c:
            s = k[0].full_simplify()*mu**k[1]
            pretty_print(s.factor())

def print_e2(elem):
    coeff = elem.coefficients(e)
    for i in range(len(coeff)):
        print(e^coeff[i][1], ":")
        c = coeff[i][0].coefficients(mu)
        for k in c:
            s = k[0].full_simplify()*mu**k[1]
            pretty_print(s.factor().full_simplify())


# create terms

def nabla_S_sq_terms(S, v, n):
    """
    Returns the 3 terms of ∇_μ Σ ∇^μ Σ*:
    |∂_μ Σ|², 
    -i (-∂_μ Σ [v, Σ*] + [v, Σ]∂_μ Σ*  ), 
    [v, Σ][v, Σ*]
    """

    dS = diff(S, x) # d_mu Sigma
    dSct = diff(S.C.T, x) #d_mu Sigma* 
    COM = v*S - S*v # [v_mu, Sigma]
    
    term1 = mat_prep(dS*(dSct), n)
    term2 = -I*mat_prep(dS*(-COM.C.T) + COM*dSct, n)
    term3 = mat_prep(COM*COM.C.T, n)
    
    return term1, term2, term3

def tr_nabla_sq(S, v, n):
    """
    Tr{∇_μ Σ ∇^μ Σ*}
    """
    term1, term2, term3 = nabla_S_sq_terms(S, v, n)

    r1 = term1.trace().full_simplify()
    r2 = term2.trace().full_simplify()
    r3 = term3.trace().full_simplify()

    r = (r1 + r2 + r3).full_simplify()
    return r