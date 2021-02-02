import numpy as np
import pandas as pd
import sys
sys.path.append('.')
import warnings
from utils import dataframe2MultiViewMatrix
from nonconvex_ALM_MRMLasso import nonconvex_ALM_MRMLasso
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
# from MRMLasso.utils import dataframe2MultiViewMatrix
warnings.filterwarnings('ignore')

Lasso_para = {
    'lambdaR' : 100 ,
    'lambdaS' : 0.01
}

gradient_size = 0.1

for data_num in range(1, 2):

    # test data
    test_ori = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    # training data
    train_ori = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))

    X_train = train_ori.drop(['Label'] , axis=1)
    y_train = train_ori['Label']
    X_test = test_ori.drop(['Label'] , axis=1)
    y_test = test_ori['Label']

    num_fea = [7 , 5 , 14 , 315 , 29 , 1271 , 280]

    _, randomSelection_X_train , _, randomSelection_y_train = train_test_split(X_train , y_train , test_size=gradient_size , stratify=y_train , random_state=10)

    # train_data = dataframe2MultiViewMatrix(X_train , num_fea)
    # test_data = dataframe2MultiViewMatrix(X_test , num_fea)
    # train_label = y_train.values
    # test_label = y_test.values
    train_data = dataframe2MultiViewMatrix(randomSelection_X_train , num_fea)
    train_label = randomSelection_y_train.values

    Beta , W , Betav_iters , Wv_iters , Ldual = nonconvex_ALM_MRMLasso(randomSelection_X_train.shape[0] , len(num_fea) , train_data , train_label , Lasso_para)
    joblib.dump(Beta,
                '/home/huxinhou/WorkSpace_BR/Multi-view/MRMLasso/result/pickle/test_algorithm/' +
                'Beta_views_{}_R_{}_S_{}_size_{}.plk'.format(len(num_fea) , Lasso_para['lambdaR'] , Lasso_para['lambdaS'] , gradient_size))
    joblib.dump(W,
                '/home/huxinhou/WorkSpace_BR/Multi-view/MRMLasso/result/pickle/test_algorithm/' +
                'W_views_{}_R_{}_S_{}_size_{}.plk'.format(len(num_fea), Lasso_para['lambdaR'], Lasso_para['lambdaS'],
                                                             gradient_size))
    joblib.dump(Ldual,
                '/home/huxinhou/WorkSpace_BR/Multi-view/MRMLasso/result/pickle/test_algorithm/' +
                'Ldual_views_{}_R_{}_S_{}_size_{}.plk'.format(len(num_fea), Lasso_para['lambdaR'], Lasso_para['lambdaS'],
                                                          gradient_size))

    print("Done..")
