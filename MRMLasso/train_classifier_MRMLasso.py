import sys
sys.path.append('.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression , Lasso
from sklearn.externals import joblib
from utils import getBetaList
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

Lasso_para = {
    'lambdaR' : 100 ,
    'lambdaS' : 0.01
}

gradient_size = 0.1

num_fea = [7 , 5 , 14 , 315 , 29 , 1271 , 280]

Beta_MRMLasso = joblib.load('/home/huxinhou/WorkSpace_BR/Multi-view/MRMLasso/result/pickle/test_algorithm/' +
                'Beta_views_{}_R_{}_S_{}_size_{}.plk'.format(len(num_fea) , Lasso_para['lambdaR'] , Lasso_para['lambdaS'] , gradient_size))

W_MRMLasso = joblib.load('/home/huxinhou/WorkSpace_BR/Multi-view/MRMLasso/result/pickle/test_algorithm/' +
                'W_views_{}_R_{}_S_{}_size_{}.plk'.format(len(num_fea), Lasso_para['lambdaR'], Lasso_para['lambdaS'],
                                                             gradient_size))

Beta_List = getBetaList(Beta_MRMLasso , num_fea)
sorted_Beta_List_index = np.argsort(-np.array(Beta_List))
selected_MRMLasso_coef_index_list = [i for i in range(len(Beta_List)) if Beta_List[i] > 0]
selected_sorted_Beta_List_index = sorted_Beta_List_index[:len(selected_MRMLasso_coef_index_list)]

# sta = 0
# selected_view_fea_num = []
# for v in range(len(num_fea)):
#     cur_view_selected_feature_list =  [i for i in selected_sorted_Beta_List_index if (i >= sta and i < sta + num_fea[v])]
#     sta += num_fea[v]
#     selected_view_fea_num.append(len((cur_view_selected_feature_list)))

topK = [5 , 10 , 15 , 20 , 25 , 30 , 35 , 40 , 45 , 50 , 60 , 70 , 80 , 90 , 100 , len(selected_MRMLasso_coef_index_list)]

for data_num in range(1 , 2):

    # test data
    test_ori = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    # training data
    train_ori = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))

    X_train = train_ori.drop(['Label'], axis=1)
    y_train = train_ori['Label']
    X_test = test_ori.drop(['Label'], axis=1)
    y_test = test_ori['Label']

    # baseline
    lr_base = LogisticRegression(n_jobs=-1)
    lr_base.fit(X_train , y_train)

    y_predict_base = lr_base.predict_proba(X_test)[:,1]
    auc_base = roc_auc_score(y_test , y_predict_base)

    # normal Lasso
    lasso = Lasso(alpha=0.001)
    lasso.fit(X_train , y_train)
    coef = lasso.coef_
    selected_normal_lasso_coef_index_list = [i for i in range(len(coef)) if coef[i] > 0]
    normal_Lasso_fea_num = len(selected_normal_lasso_coef_index_list)

    X_train_NormalLasso = X_train.iloc[:,selected_normal_lasso_coef_index_list]
    X_test_NormalLasso = X_test.iloc[:,selected_normal_lasso_coef_index_list]

    lr_NormalLasso = LogisticRegression(n_jobs=-1)
    lr_NormalLasso.fit(X_train_NormalLasso , y_train)

    y_predict_NormalLasso = lr_NormalLasso.predict_proba(X_test_NormalLasso)[:,1]
    auc_NormalLasso = roc_auc_score(y_test , y_predict_NormalLasso)

    # use MRMLasso
    X_train_MRMLasso = X_train.iloc[:,selected_MRMLasso_coef_index_list]
    X_test_MRMLasso = X_test.iloc[:,selected_MRMLasso_coef_index_list]

    lr_MRMLasso = LogisticRegression(n_jobs=-1)
    lr_MRMLasso.fit(X_train_MRMLasso , y_train)

    y_predict_MRMLasso = lr_MRMLasso.predict_proba(X_test_MRMLasso)[:,1]
    auc_MRMLasso = roc_auc_score(y_test , y_predict_MRMLasso)

    # # use MRMLasso to train classifier for each view
    # df_train_cur_view_stacking_input = pd.DataFrame(np.zeros((X_train.shape[0] , len(num_fea))))
    # df_test_cur_view_stacking_input = pd.DataFrame(np.zeros((X_test.shape[0], len(num_fea))))
    # delete_col_list = []
    # sta = 0
    # for v in range(len(num_fea)):
    #     cur_view_selected_feature_list =  [i for i in selected_MRMLasso_coef_index_list if (i >= sta and i < sta + num_fea[v])]
    #     sta += num_fea[v]
    #     if len(cur_view_selected_feature_list) == 0 :
    #         df_train_cur_view_stacking_input.iloc[:, v] = np.zeros((X_train.shape[0] , 1))
    #         df_test_cur_view_stacking_input.iloc[:, v] = np.zeros((X_test.shape[0] , 1))
    #         delete_col_list.append(v)
    #         continue
    #     X_train_MRMLasso_each_view = X_train.iloc[:,cur_view_selected_feature_list]
    #     X_test_MRMLasso_each_view = X_test.iloc[:,cur_view_selected_feature_list]
    #     lr_MRMLasso_cur_view = LogisticRegression(n_jobs=-1)
    #     lr_MRMLasso_cur_view.fit(X_train_MRMLasso_each_view , y_train)
    #     df_train_cur_view_stacking_input.iloc[:,v] = lr_MRMLasso_cur_view.predict_proba(X_train_MRMLasso_each_view)[:,1]
    #     df_test_cur_view_stacking_input.iloc[:, v] = lr_MRMLasso_cur_view.predict_proba(X_test_MRMLasso_each_view)[:,1]
    # df_train_cur_view_stacking_input = df_train_cur_view_stacking_input.drop(delete_col_list , axis=1)
    # df_test_cur_view_stacking_input = df_test_cur_view_stacking_input.drop(delete_col_list, axis=1)
    #
    # lr_MRMLasso_stacking = LogisticRegression(n_jobs=-1 , C = 0.1)
    # lr_MRMLasso_stacking.fit(df_train_cur_view_stacking_input , y_train)
    # y_predict_MRMLasso_stacking = lr_MRMLasso_stacking.predict_proba(df_test_cur_view_stacking_input)[:,1]
    # auc_MRMLasso_stacking = roc_auc_score(y_test , y_predict_MRMLasso_stacking)

    # test MRMLasso topK
    auc_MRMLasso_df = pd.DataFrame(np.zeros((1 , len(topK))) , index=['auc'] , columns=topK)
    for k in topK:
        topK_MRMLasso_selected_coef_list = selected_sorted_Beta_List_index[:k]
        X_train_MRMLasso = X_train.iloc[:, topK_MRMLasso_selected_coef_list]
        X_test_MRMLasso = X_test.iloc[:, topK_MRMLasso_selected_coef_list]
        lr_MRMLasso = LogisticRegression(n_jobs=-1)
        lr_MRMLasso.fit(X_train_MRMLasso, y_train)
        y_predict_MRMLasso = lr_MRMLasso.predict_proba(X_test_MRMLasso)[:, 1]
        auc_MRMLasso = roc_auc_score(y_test, y_predict_MRMLasso)
        auc_MRMLasso_df.loc['auc' , k] = auc_MRMLasso

    auc_MRMLasso_df.to_csv('/home/huxinhou/WorkSpace_BR/Multi-view/MRMLasso/result/' + 'MRMLasso_topK_auc_R_{}_S_{}.csv'.format(Lasso_para['lambdaR'] , Lasso_para['lambdaS']) )




