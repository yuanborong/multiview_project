import numpy as np

def computeLpri1(m , s ,Beta , W , lambdaR , lambdaS , train_label , train_data):

    pri_terms1 = []

    # term1
    tmp = np.zeros((m , 1))
    for v in range(s):
        vec = train_data[v] * Beta[v]
        tmp = tmp + np.multiply(W[:,v] , vec)
    term = np.linalg.norm(train_label - tmp , ord=2)
    term = 0.5 / m * term * term
    pri_terms1.append(term)

    # term2
    term = 0
    n = 0
    for v in range(s):
        term = term + np.linalg.norm(Beta[v] , ord=1)
        n = n + len(Beta[v])
    term = term * lambdaS / s
    pri_terms1.append(term)

    # term3
    Wr = np.around(W , 4)
    term = lambdaR * np.linalg.matrix_rank(Wr)
    pri_terms1.append(term)

    values = np.sum(pri_terms1)

    return values , pri_terms1