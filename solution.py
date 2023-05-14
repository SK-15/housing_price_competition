# -*- coding: utf-8 -*-
"""
Created on Sat May 13 20:26:18 2023

@author: saura
"""

import pandas as pd
import numpy as np 
import glob
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['image.cmap'] = 'tab20'


if __name__ == '__main__':
    
    path = r'C:\Users\saura\Python Programs\Data_Science_competition\House_price_new\data'
    csv_files = glob.glob(os.path.join(path,"*.csv"))
    data_dict = {}
    
    csv_files = [s.replace(path+'\\','') for s in csv_files]
    csv_files = [s.replace('.csv','') for s in csv_files]
    
    for csv in csv_files:
        data_dict[csv] = pd.read_csv(rf'{path}\{csv}.csv',parse_dates=True,index_col=0)
        
    test = data_dict['test']
    train = data_dict['train']
    
    # train = train.dropna()
    # na_cols =  train.isna().sum()
    train = train.drop(['PoolQC','Fence','MiscFeature','FireplaceQu','Alley','LotFrontage'],axis=1)
    train = train.dropna()
    
    types_lst = train.dtypes
    types_lst = list(types_lst[types_lst=='object'].index)
    
    train_cat = pd.get_dummies(train[types_lst])
    train = pd.concat([train,train_cat],axis=1)
    train = train.drop(types_lst,axis=1)
    
    y = train['SalePrice']
    X = train.drop(['SalePrice'],axis=1)
    
    scaler = StandardScaler() 
    scaler.fit(X)
    X = scaler.transform(X)
    # X = pd.to
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestRegressor()
    model.fit(X_train,y_train)
    
    y_pred = model.predict(X_test)
    
    score1 = mean_squared_error(y_test,y_pred)
    
    