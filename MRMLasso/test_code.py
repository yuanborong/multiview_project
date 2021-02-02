import sys
sys.path.append('.')
import numpy as np
from MRMLasso.computeLpri1 import computeLpri1
from MRMLasso.computeLpri2 import computeLpri2
from MRMLasso.computeLdual import computeLdual
from MRMLasso.nonconvex_ALM_MRMLasso import nonconvex_ALM_MRMLasso

m = 2
s = 3
nv = [3 , 3 , 3]
lambdaR = 1
lambdaS = 1
rho = 1
mu = 1
xi = 1
Lasso_para = {
    'lambdaR' : 100 ,
    'lambdaS' : 0.5
}

train_data = [np.mat([[1 , 2 , 3] , [1 , 2 , 3]]) , np.mat([[2 , 5 , 9] , [2 , 5 , 9]]) , np.mat([[1 , 2 , 3] , [1 , 2 , 3]])]
train_label = [0 , 1]
# W = m * s
W = np.mat([[0.1 , 0.5 , 0.4] , [0.5 , 0.2 , 0.3]])
# Beta each element is vth view's Betav , each element is a matrix which size is (nv * 1)
# Beta = [np.mat(np.random.uniform(0 , 1 , (nv[0] , 1))) , np.mat(np.random.uniform(0 , 1 , (nv[1] , 1))) , np.mat(np.random.uniform(0 , 1 , (nv[2] , 1)))]
Beta = [np.mat([0.1 , 0.2 , 0.3]).T , np.mat([0.1 , 0.2 , 0.3]).T , np.mat([0.1 , 0.2 , 0.3]).T]
# J = m * s
J = np.mat([[0.9 , 0.1 , 0.2] , [0.4 , 0.6 , 0.1]])
# Uv = m * nv = W[:,v] * Beta[v].T
U = []
for j in range(s):
    # x = np.reshape(W[:,j] , (2 , 1))
    # y = np.reshape(Beta[j] , (1 , 3))
    # U.append(x * y)
    U.append(np.mat(np.dot(W[:,j] , Beta[j].T)))

# Lagrange multipliers P,Q,R
# Pv = m * nv
P = []
for j in range(s):
    P.append(np.mat([[1, 1, 1], [1, 1, 1]]))
# Q = m * s
Q = np.mat([[1 , 5 , 4] , [0.5 , 3 , 0.3]])
# R = m * s
R = np.mat([[0.2 , 0.1 , 0.2] , [0.1 , 0.6 , 0.1]])

# test computeLpri1
# values1 , pri_terms1 = computeLpri1(m , s , Beta , W , 1 , 1 , train_label , train_data)

# test computeLpri2
# values2 , pri_terms2 = computeLpri2(m , s , Beta , U , J , lambdaR , lambdaS , train_label , train_data)

# test computeLdual
# values3 , dual_terms = computeLdual(m , s , pri_terms2 , Beta , W , U , J , P , Q , R , rho , mu , xi)

Beta , W = nonconvex_ALM_MRMLasso(m , s , train_data , train_label , Lasso_para)

print(Beta)
print(W)
# print(values1 , pri_terms1)
# print(values2 , pri_terms2)
# print(values3 , dual_terms)