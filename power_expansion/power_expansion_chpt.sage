# Hack
# Needed to take conjugate of pi functions
# Do not know of a better way

def Id(self, x): return self(x)



POW = lambda A, n : matrix.identity(A.dimensions()[0]) if (n == 0) else A * POW(A, n-1)
EXP = lambda A, n : sum([POW(A, i)/factorial(i) for i in range(n+1)])


acom = lambda A1, A2 : A1 * A2 + A2 * A1
com = lambda A1, A2 : A1 * A2 - A2 * A1


# Project matrices onto basis of pauli matrices
def proj(A, s):
    n = A.nrows()
    return [(A.trace()/n).full_simplify(), *[((A*si).trace()/2).full_simplify() for si in s]]


# Helper functions

def mat_series(mat, x, n):
    d = mat.dimensions()
    for i in range(0, d[0]):
        for j in range(0, d[1]):
            mat[i, j] = mat[i, j].series(x, n+1).truncate()

def mat_simp(mat):
    d = mat.dimensions()
    for i in range(0, d[0]):
        for j in range(0, d[1]):
            mat[i, j] = mat[i, j].full_simplify()
            mat[i, j] = mat[i, j].trig_reduce()

def mat_prep(mat, n):
    mat_series(mat, e, n)
    # mat_simp(mat)
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