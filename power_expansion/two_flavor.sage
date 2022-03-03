load("power_expansion_chpt.sage")
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
p = vector([
    function("pi"+str(i), latex_name="\\pi_"+str(i), conjugate_func=Id)(x) for i  in range(1, len(s)+1)
])
pi_s = e * sum([s[i]*p[i] for i in range(len(s))])



A_a = one*cos(a/2) + I * s1 * sin(a/2)
U = lambda n: EXP(I * pi_s/2, n)
SIGMA = lambda n: mat_prep(A_a * U(n) * U(n) * A_a, n)
