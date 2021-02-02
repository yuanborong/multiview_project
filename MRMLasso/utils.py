import numpy as np
import pandas as pd
import sys
sys.path.append('.')
import warnings
from tkinter import _flatten
warnings.filterwarnings('ignore')

def dataframe2MultiViewMatrix(dataframe , num_fea):

    mutlti_matrix = []
    sta = 0
    for v in range(len(num_fea)):
        curViewDataframe = dataframe.iloc[:,sta:sta+num_fea[v]]
        mutlti_matrix.append(np.mat(curViewDataframe.values))
        sta = sta + num_fea[v]

    return mutlti_matrix


def getBetaList(Beta , num_fea):

    Beta_List = []
    for v in range(len(num_fea)):
        for i in range(num_fea[v]):
            Beta_List.append(float(Beta[v][i]))

    return Beta_List
