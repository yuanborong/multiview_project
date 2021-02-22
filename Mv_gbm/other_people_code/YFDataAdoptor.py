"""
【输入】
    1. 需要转换的list文件路径
    2. 转换完成后保存的路径
    
读取并转换师兄处理后的数据
    1. 删除 day字段
    2. 重置列名
    3. 删除空列
"""
import sys
import pandas as pd

def deleteNullCol(samples):
    markedIndex = []
    for column in samples[: -1]:
        if (samples[column].nunique() == 1):
            markedIndex.append(column)
    samples.drop(markedIndex, axis=1, inplace=True)
    return samples

def cvtFeatureName(samples):
    demoNameTable = { 
        "1": "Age",      "2": "Hispanic",  "3": "Race", 
        "4": "Sex" 
    }
    vitalNameTable = {
        "1": "Height",   "2": "Weight",    "3": "BMI",
        "4": "Smoking",  "5": "Tobacco",   "6": "TobaccoType",
        "7": "SBP",      "8": "DBP"
    }
    featureNames = samples.columns
    demoNames, vitalNames, labNames, medNames, ccsNames, pxNames = [], [], [], [], [], []
    for name in featureNames:
        prefix = name[: 2]
        if prefix == "de":                       # 32
            index = name[4]
            newName = "DEMO_" + demoNameTable[index]
            if len(name) > 5:
                value = name[5:]
                newName = newName + "_" + value
            demoNames.append(newName)
        elif prefix == "vi":                     # 32
            index = name[5]
            value = name[6:]
            newName = "VITAL_" + vitalNameTable[index]
            if value != "":
                newName = newName + "_" + value
            vitalNames.append(newName)
        elif prefix == "la":                     # 817
            index = name[3:]
            newName = "LAB_" + index
            labNames.append(newName)
        elif prefix == "cc":                     # 280
            index = name[3:]
            newName = "CCS_" + index
            ccsNames.append(newName)
        elif prefix == "px":                     # 15606
            index = name[2:]
            newName = "PX_" + index
            pxNames.append(newName)
        elif prefix == "me":                     # 15539
            index = name[3:]
            newName = "MED_" + index
            medNames.append(newName)
    newFeatureName = demoNames + vitalNames + labNames + ccsNames + pxNames + medNames + ["Label"]
    return newFeatureName
# evn = sys.argv[1]
year = sys.argv[1]
# constant = CONSTANT(evn)
samplesPath = "/panfs/pfs.local/work/liu/xzhang_sta/yuanborong/data/df_data/pre_48h/data-{}.pkl".format(year) # 你用YFDataGenerate.py生成的YFSample-****.pkl所在的路径
savePath = "/panfs/pfs.local/work/liu/xzhang_sta/yuanborong/data/df_data/pre_48h/no_null_col_data-{}.pkl".format(year)    # 处理后数据保存的位置
samples = pd.read_pickle(samplesPath)
# 删除days列
samples.drop(["days"], axis=1, inplace=True)
# 重置列名
samples.columns = cvtFeatureName(samples)
# 删除空列
deletedNullColSamples = deleteNullCol(samples)
# 保存文件
deletedNullColSamples.to_pickle(savePath)
print("saved data in {}".format(savePath))