import numpy as np

def computeLdual(m , s , pri_terms2 , Beta , W , U , J , P , Q , R , rho , mu , xi):

    dual_terms = []

    # term1-3
    dual_terms = dual_terms + pri_terms2

    # term4-5
    term = 0
    term1 = 0
    for v in range(s):
        # m * nv
        mat = -U[v] + W[:,v] * Beta[v].T
        term = term + np.sum(np.diag(P[v].T * mat))
        term1 = term1 + rho / 2 * np.linalg.norm(mat , ord='fro')
    dual_terms.append(term)
    dual_terms.append(term1)

    # term6
    mat = Q - mu * W
    term = 0
    for i in range(m):
        for v in range(s):
            if mat[i , v] > 0:
                term = term + mat[i , v] * mat[i , v] - Q[i , v] * Q[i , v]
            else:
                term = term - Q[i , v] * Q[i , v]
    term = term * 0.5 / mu
    dual_terms.append(term)

    # term8-9
    mat = J - W
    term = np.trace(R.T * mat)
    term1 = xi / 2 * np.linalg.norm(mat , ord='fro')
    dual_terms.append(term)
    dual_terms.append(term1)

    values = np.sum(dual_terms)

    return values , dual_terms
