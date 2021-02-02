import sys
sys.path.append('.')
import numpy as np
from computeLpri1 import computeLpri1
from computeLpri2 import computeLpri2
from computeLdual import computeLdual

def nonconvex_ALM_MRMLasso(m , s , train_data , train_label , Lasso_para):

    np.random.seed(10)
    tol = 1e-4

    lambdaR = Lasso_para['lambdaR']
    lambdaS = Lasso_para['lambdaS']

    spliced_data = []
    for v in range(s):
        spliced_data.append(train_data[v])
    spliced_data = np.mat(np.concatenate(spliced_data , axis=1))

    # initializing W , beta , lagrange multipliers P,Q,R , /rho , /mu , /xi
    num_feas = []
    # Betapre[v] is vth view's Beta coef whose size is (1 , nv)
    Betapre = []
    for v in range(s):
        Betapre.append(np.mat(np.random.uniform(0 , 1 , (train_data[v].shape[1] , 1))))
        num_feas.append(train_data[v].shape[1])
    all_features = np.sum(num_feas)
    randW = np.mat(np.random.uniform(0 , 1 , (m , s)))
    Wpre = np.mat(np.zeros((m , s)))
    W = np.mat(np.zeros((m , s)))
    for i in range(m):
        Wpre[i,:] = randW[i,:] / np.sum(randW[i,:])
    P = []
    for v in range(s):
        P.append(np.mat(np.zeros((m , num_feas[v]))))
    Q = np.mat(np.zeros((m , s)))
    R = np.mat(np.zeros((m , s)))
    rho = 1
    mu = 1
    xi = 1
    step = 1.1
    rho_bar = rho * 1e4
    mu_bar = mu * 1e6
    xi_bar = xi * 1e6

    # compute Gamma and J
    while 1:
        UU , SS , VV = np.linalg.svd(Wpre , full_matrices=False)
        VV = VV.T
        # diagS = np.diag(SS)
        diagS = SS
        svp = np.count_nonzero(diagS > (lambdaR / xi))
        if svp < 1:
            # due to the zero matrix
            if xi == xi_bar:
                print('error , mu is the max , svp is zero')
                Jpre = np.mat(np.zeros((m , s)))
                break
            xi = np.min([xi * step , xi_bar])
            continue
        sv = s
        if svp < sv:
            sv = np.min([svp + 1 , s])
        else:
            sv = np.min([svp + np.round(0.05 * s) , s])
        Jpre = UU[: , 0:svp] * np.diag(diagS[0:svp] - (lambdaR / xi)) * VV[:,0:svp].T
        break

    Upre = []
    for v in range(s):
        Upre.append(np.mat(np.zeros((m , num_feas[v]))))

    Lpri1 = []
    Lpri2 = []
    Ldual = []

    residual1_list = []
    residual2_list = []
    residual4_list = []
    residual5_list = []
    residual6_list = []
    residual7_list = []
    stop_list = []
    Ldual.append(np.inf)
    for v in range(s):
        residual1_list.append([])
        residual4_list.append([])
        residual6_list.append([])
        residual7_list.append([])

    # alternating direction multipliers of methods and augmented lagrange
    # U -> beta -> W -> Gamma and J
    iter = 1
    Betav_iters = []
    Wv_iters = []
    Uv_iters = []
    for v in range(s):
        Betav_iters.append([])
        Wv_iters.append([])
        Uv_iters.append([])

    while iter < 10000 :

        # solve U
        C = []
        spliced_P = []
        for v in range(s):
            C.append(Wpre[:,v] * Betapre[v].T)
            spliced_P.append(P[v])
        C = np.mat(np.concatenate(C , axis=1))
        spliced_P = np.mat(np.concatenate(spliced_P , axis=1))

        spliced_U = np.mat(np.zeros((m , all_features)))
        for i in range(m):
            x_i = spliced_data[i,:].T
            q = train_label[i] * x_i + m * spliced_P[i,:].T + rho * C[i,:].T
            tmp = q.T * x_i / (rho * m + x_i.T * x_i)
            spliced_U[i,:] = 1 / (rho * m) * (q - float(tmp[0,0]) * x_i).T

        # solve Beta S_(lambdaS / (s * d * rho))(D / (d * rho))
        sta = 0
        U = []
        Beta = []
        for v in range(s):
            U.append(spliced_U[:,sta:sta+num_feas[v]])
            sta = sta + num_feas[v]
            D = (rho * U[v] - P[v]).T * Wpre[:,v]
            d = Wpre[:,v].T * Wpre[:,v]
            # mat = nv * 1
            mat = D / (rho * d)
            par = lambdaS / (s * d * rho)
            Beta.append(np.mat(np.zeros((num_feas[v] , 1))))
            for j in range(num_feas[v]):
                # print('mat[j] :' + str(mat[j]) + ' ; par : ' + str(par) + ' ; ')
                if mat[j] > par:
                    Beta[v][j] = mat[j] - par
            Betav_iters[v].append(Beta[v])
            Uv_iters[v].append(U[v])

        # solve W
        A = xi * Jpre + R
        for v in range(s):
            par = float(rho * Beta[v].T * Beta[v] + xi)
            Bv = (rho * U[v] - P[v]) * Beta[v]
            cond1 = 1 / par * (A[:,v] + Bv)
            cond2 = 1 / (par + mu) * (A[:,v] + Bv + Q[:,v])
            cond3 = 1 / mu * Q[:,v]
            for i in range(m):
                if cond1[i] >= cond3[i] and cond2[i] >= cond3[i]:
                    W[i,v] = cond1[i]
                else:
                    W[i,v] = cond2[i]
        # normalize
        W = W / np.sum(W , axis=1)
        for v in range(s):
            Wv_iters[v].append(W[:,v])

        # solve GA
        GA = W - 1 / mu * Q
        for i in range(m):
            for v in range(s):
                if GA[i,v] < 0:
                    GA[i,v] = 0

        # solve J
        while 1:
            UU , SS , VV = np.linalg.svd((W - R / xi) , full_matrices=False)
            VV = VV.T
            diagS = SS
            svp = np.count_nonzero(diagS > (lambdaR / xi))
            if svp < 1 :
                # due to the zero matrix
                if xi == xi_bar:
                    print('error , mu is the max , svp is zero')
                    J = np.mat(np.zeros((m, s)))
                    break
                xi = np.min([xi * step, xi_bar])
                continue
            if svp < sv:
                sv = np.min([svp + 1 , s])
            else:
                sv = np.min([svp + np.round(0.05 * s) , s])
            J = UU[:,0:svp] * np.diag(diagS[0:svp] - (lambdaR / xi)) * VV[:,0:svp].T
            break

        # update lagrange multipliers P Q R
        Ppre = []
        for v in range(s):
            Ppre.append(P[v])
            P[v] = P[v] + rho * (-U[v] + W[:,v] * Beta[v].T)
            for i in range(m):
                if Q[i,v] - mu * W[i,v] <= 0:
                    Q[i,v] = 0
                else:
                    Q[i,v] = Q[i,v] - mu * W[i,v]
        R = R + xi * (J - W)

        # the function value varies
        values1 , pri_term1 = computeLpri1(m , s , Beta , W , lambdaR , lambdaS , train_label , train_data)
        Lpri1.append(values1)
        values2 , pri_term2 = computeLpri2(m , s , Beta , U , J , lambdaR , lambdaS , train_label , train_data)
        Lpri2.append(values2)
        values3 , dual_terms = computeLdual(m , s , pri_term2 , Beta , W , U , J , P , Q , R , rho , mu , xi)
        Ldual.append(values3)

        # stop conditions
        stop_i = np.zeros((1 , 3 * s + 2))
        p = -1
        stop = 0
        err_rel = 1e-1
        err_abs = 1e-4
        residual1 = []
        for v in range(s):
            p = p + 1
            residual1.append(np.linalg.norm(U[v] - W[:,v] * Beta[v].T , ord='fro'))
            err_pri = np.sqrt(m * num_feas[v]) * err_abs + \
                      err_rel * np.max([np.linalg.norm(U[v] , ord='fro') , np.linalg.norm(W[:,v] * Beta[v].T , ord='fro')])
            if residual1[v] <= err_pri:
                stop = stop + 1
                stop_i[0,p] = 1
            residual1_list[v].append(residual1[v])

        p = p + 1
        residual2 = np.linalg.norm(W -J , ord=2)
        err_pri = np.sqrt(m * s) * err_abs + err_rel * np.max([np.linalg.norm(W , ord='fro') , np.linalg.norm(J , ord='fro')])
        if residual2 <= err_pri:
            stop = stop + 1
            stop_i[0,p] = 1
        residual2_list.append(residual2)

        mat = W - Wpre
        residual4 = []
        for v in range(s):
            p = p + 1
            residual4.append(np.linalg.norm((Ppre[v] + rho * Wpre[:,v] * Beta[v].T).T * mat[:,v]  , ord=2))
            err_dual = np.sqrt(num_feas[v]) * err_abs + err_rel * np.linalg.norm(P[v].T * W[:,v] )
            if residual4[v] <= err_dual:
                stop = stop + 1
                stop_i[0,p] = 1
            residual4_list[v].append(residual4)

        mat = J - Jpre
        p = p + 1
        residual5 = np.linalg.norm(-xi * mat , ord='fro')
        err_dual = np.sqrt(m * s) * err_abs + err_rel * np.linalg.norm(Q + R , ord=2)
        if residual5 <= err_dual:
            stop = stop + 1
            stop_i[0,p] = 1
        residual5_list.append(residual5)

        residual6 = []
        for v in range(s):
            p = p + 1
            residual6.append(
                np.linalg.norm(-rho * (W[:,v] * Beta[v].T - Wpre[:,v] * Betapre[v].T) , ord='fro')
            )
            err_dual = np.sqrt(m * num_feas[v]) * err_abs + err_rel * np.linalg.norm(P[v] , ord='fro')
            if residual6[v] <= err_dual:
                stop = stop + 1
                stop_i[0,p] = 1
            residual6_list[v].append(residual6)
        stop_list.append(stop_i)

        residual7 = []
        for v in range(s):
            residual7.append(
                np.linalg.norm((U[v] - Upre[v]) , ord='fro')
            )
            residual7_list[v].append(residual7)

        # terminal
        if stop == 2 + 3 * s or iter == 200:
            break

        if np.abs(Ldual[-1] - Ldual[-2]) < tol:
            break

        # update rho , mu , xi
        if np.abs(dual_terms[4]) / rho > 1e-5:
            rho = np.min([rho * step , rho_bar])
        mu = np.min([mu * step , mu_bar])
        xi = np.min([xi * step , xi_bar])

        # variable updating
        Upre = U
        Betapre = Beta
        Wpre = W
        # GApre = GA
        Jpre = J

        print('Iteration: ' + str(iter) + ' ; dual function value: ' + str(np.abs(Ldual[-1] - Ldual[-2])) + ' ;')
        iter = iter + 1

    return Beta , W , Betav_iters , Wv_iters , Ldual








