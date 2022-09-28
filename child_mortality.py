# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 19:02:06 2022

@author: h
"""

from sklearn.linear_model import LinearRegression
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

gm = pd.read_csv('gm_2008_region.csv')
X = gm[['population','fertility','BMI_male','GDP','BMI_female','life']]
y = gm['child_mortality']
print(X.head())
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=101)
reg = LinearRegression()
reg.fit(X_train, y_train)
child_mort = float(np.round(reg.predict([[58372889, 2.0, 25, 8600, 125, 75]]), 2))
print("""\n\nChild mortality for a country with 
      Population: 58372889 
      Fertility: 2 
      BMI Male: 25 
      GDP: 8600 
      BMI Female: 125 
      Life Exp: 75 \nis {}""".format(child_mort))

      
pickle.dump(reg, open('model.pickle', 'wb'))