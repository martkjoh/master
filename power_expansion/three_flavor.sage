load("power_expansion_chpt.sage") 
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
var("mu", latex_name="\\mu_I", domain="real")
var("a", latex_name="\\alpha", domain="real")

# e A_mu
var("eA", latex_name="e \\mathcal A_\\mu", domain="real")


# chi with mass

# var("dm", latex_name="\\Delta m", domain="real")
# var("mm", latex_name="\\bar m", domain="real")
# var("B0", latex_name="B_0", domain="real")

# chi = (mm^2 * one + dm^2 * s3)


# pi_a tau_a

# pi_s = e * sum([s[i]*p[i] for i in range(len(s))])



# A_a = one*cos(a/2) + I * s1 * sin(a/2)
# U = lambda n: EXP(I * pi_s/2, n)
# SIGMA = lambda n: mat_prep(A_a * U(n) * U(n) * A_a, n)


