"""
Created on Wed Nov 27 11:27:08 2019

@author: Shriraj-PC
"""
#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime


#reading the data
dataframe = pd.read_csv('BrentOilPrices.csv')
X = dataframe.iloc[:,:-1].values
y = dataframe.iloc[:,1].values

#declaring a matrix 
x = [[0] * 3] * len(X)

#converting the date to separate entities
for i in range(len(X)):
        dt = str(X[i])
        dt = dt.strip('[]\'')
        #print(dt)
        x[i] = re.split("\s", dt)
        x[i][1] = x[i][1].strip(',')

#Converting a 2d array to np array Ques- try to declate np array at first       
x = np.array(x)

#Conversion of Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder        
label_X = LabelEncoder()
x[:,0] = label_X.fit_transform(x[:,0])
one_ht_enc = OneHotEncoder(categorical_features=[0])
x = one_ht_enc.fit_transform(x).toarray()

#Avoiding the dummy variable
x = x[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting into multiple linear reg model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

#Predicing the test results
y_pred = reg.predict(X_test)

#Accuracy 
from sklearn import metrics
print('Accuracy = '+str(metrics.r2_score(y_test,y_pred)*100))

############################################################################################
#making an optimum model using backward elimination and adding an extra col of ones for it
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((len(x),1)).astype(int), values = x , axis = 1)
#We are adding a column of ones to x because of the coefficient b0 that we didn't
#consider earlier Eg- y = b0 + b1*x1+ b2*x2.... so x0 in this case is 1
#11111111111111111111111111111111111111111111111111111
x_opt = x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
reg_ols = sm.OLS(endog = y, exog = x_opt).fit() 
reg_ols.summary() 

#removed x9 or 9
x_opt = x[:,[0,1,2,3,4,5,6,7,8,10,11,12,13]]
reg_ols = sm.OLS(endog = y, exog = x_opt).fit() 
reg_ols.summary() 

#removed x11 or 12
x_opt = x[:,[0,1,2,3,4,5,6,7,8,10,11,13]]
reg_ols = sm.OLS(endog = y, exog = x_opt).fit() 
reg_ols.summary()

#removed x2 or 2
x_opt = x[:,[0,1,3,4,5,7,8,9,10,13]]
reg_ols = sm.OLS(endog = y, exog = x_opt).fit() 
reg_ols.summary()  

#removed x7 or 9
x_opt = x[:,[0,1,3,4,5,7,8,10,13]]
reg_ols = sm.OLS(endog = y, exog = x_opt).fit() 
reg_ols.summary()  

#removed x6 or 8
x_opt = x[:,[0,1,3,4,5,7,10,13]]
reg_ols = sm.OLS(endog = y, exog = x_opt).fit() 
reg_ols.summary()  

#removed x4 or 5
x_opt = x[:,[0,1,3,4,7,10,13]]
reg_ols = sm.OLS(endog = y, exog = x_opt).fit() 
reg_ols.summary()  

#removed x1 or 1
x_opt = x[:,[0,3,4,7,10,13]]
reg_ols = sm.OLS(endog = y, exog = x_opt).fit() 
reg_ols.summary()  

#removed x4 or 10
x_opt = x[:,[0,3,4,7,13]]
reg_ols = sm.OLS(endog = y, exog = x_opt).fit() 
reg_ols.summary()

#cutting to training and test
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(x_opt, y, test_size = 0.25, random_state = 0)


 
#Predicing the test results
y_pred_opt = reg_ols.predict(X_test_opt)

#Accuracy 
from sklearn import metrics
print('Accuracy = '+str(metrics.r2_score(y_test,y_pred_opt)*100))

#####################################################################################
#dECISION TREE regression
from sklearn.tree import DecisionTreeRegressor
reg_tree = DecisionTreeRegressor(random_state = 0)
reg_tree.fit(X_train, y_train)

y_pred_tree = reg_tree.predict(X_test)

#Accuracy 
from sklearn import metrics
print('Accuracy = '+str(metrics.r2_score(y_test,y_pred_tree)*100))

y_fun = reg_tree.predict([X_train[10]])
########################################################################################

#random forest regression
from sklearn.ensemble import RandomForestRegressor
reg_random_forest = RandomForestRegressor(n_estimators = 10, random_state = 0)
reg_random_forest.fit(X_train, y_train)

y_pred_tree_rand = reg_random_forest.predict(X_test)

#Accuracy 
from sklearn import metrics
print('Accuracy = '+str(metrics.r2_score(y_test,y_pred_tree_rand)*100))
