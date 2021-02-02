import numpy as np

def computeLpri2(m , s , Beta , U , J ,lambdaR , lambdaS , train_label , train_data):

    pri_term2 = []

    # term1
    tmp = np.zeros((m , 1))
    for v in range(s):
        vec = np.diag(train_data[v] * U[v].T)
        tmp = tmp + vec
    term = np.linalg.norm(train_label - tmp , ord=2)
    term = 0.5/m*term*term
    pri_term2.append(term)

    # term2
    term = 0
    n = 0
    for v in range(s):
        term = term + np.linalg.norm(Beta[v] , ord=1)
        n = n + len(Beta[v])
    term = term * lambdaS / s
    pri_term2.append(term)

    # term3
    sval = np.linalg.svd(J , compute_uv=False)
    term = lambdaR * sum(sval)
    pri_term2.append(term)

    values = sum(pri_term2)

    return values , pri_term2