import pandas as pd
import numpy as np


from utils import train_test_split, standardize, to_categorical, normalize
from utils import mean_squared_error, accuracy_score
from Multi_XGB_Model import XGBoost

def main():
    print ("-- XGBoost --")

    data = pd.read_excel('E:\\文章\\茅一段\\茅一段\\回归问题总样本.xlsx')
    id = np.array(data['POR', 'TOC', 'STL'])
    labels = np.array(data['value'])
    value = data['value']
    features = np.array(data.drop('value', 'POR', 'TOC', 'STL', axis=1))
    for i in range(len(id)):
        value = 1/(1+np.exp(value*(id[0]+1)*10+(id[1]+1)*1+(id[2]+1)*0.1)) #encode

    X = features
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    model1 = XGBoost()
    model1.fit(X_train, y_train)
    y_pred = model1.predict(X_test)

    data_por = pd.read_excel('E:\\文章\\茅一段\\茅一段\\孔隙度总样本.xlsx')
    id_por = np.array(data_por['POR'])
    data_por = np.array(data_por.drop('POR', axis=1))
    data_TOC = pd.read_excel('E:\\文章\\茅一段\\茅一段\\TOC总样本.xlsx')
    id_TOC = np.array(data_TOC['TOC'])
    data_TOC = np.array(data_TOC.drop('STL', axis=1))
    data_stl = pd.read_excel('E:\\文章\\茅一段\\茅一段\\STL总样本.xlsx')
    id_stl = np.array(data_stl['STL'])
    data_stl = np.array(data_stl.drop('STL', axis=1))

    y_por_decode = y_pred[:, :len(data_por)]
    y_TOC_decode = y_pred[:, :len(data_TOC)]
    y_stl_decode = y_pred[:, :len(data_stl)]

    X_por_train, X_por_test, y_por_train, y_por_test = train_test_split(data_por, id_por, test_size=0.5)
    X_TOC_train, X_TOC_test, y_TOC_train, y_TOC_test = train_test_split(data_TOC, id_TOC, test_size=0.5)
    X_stl_train, X_stl_test, y_stl_train, y_stl_test = train_test_split(data_stl, id_stl, test_size=0.5)

    model2 = XGBoost()
    model2.fit(X_por_train, y_por_train)
    y_por_pred = model2.predict(X_por_test)

    model3 = XGBoost()
    model3.fit(X_TOC_train, y_TOC_train)
    y_TOC_pred = model3.predict(X_TOC_test)

    model4 = XGBoost()
    model4.fit(X_stl_train, y_stl_train)
    y_stl_pred = model3.predict(X_stl_test)

if __name__ == "__main__":
    main()