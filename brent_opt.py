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
from time import strptime


#Reading the data
dataframe = pd.read_csv('BrentOilPrices.csv')
X = dataframe.iloc[:,:-1].values
y = dataframe.iloc[:,1].values

#Declaring a matrix so that we need to divide the date into month, day and year 
x = [[0] * 3] * len(X)


#Converting the date to separate month, day and year
for i in range(len(X)):
        x[i] = re.split("\s", str(X[i]).strip('[]\''))
        x[i][1] = x[i][1].strip(',')

#Converting a 2d array to np array Ques- try to declate np array at first       
x = np.array(x)

#Month convert to number
def monthConvert(x):
        for i in range(len(x)):
                x[i,0] = strptime(x[i,0],'%b').tm_mon
                
monthConvert(x) 


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)



################################################################################

#Fitting into multiple linear reg model
def multiLinReg():
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(X_train, y_train)

        #Predicing the test results
        y_pred = reg.predict(X_test)

        #Accuracy 
        from sklearn import metrics
        print('Accuracy = '+str(metrics.r2_score(y_test,y_pred)*100))


#####################################################################################
#dECISION TREE regression
def decisionTreeReg():        

        from sklearn.tree import DecisionTreeRegressor
        reg_tree = DecisionTreeRegressor(random_state = 0)
        reg_tree.fit(X_train, y_train)

        y_pred_tree = reg_tree.predict(X_test)

        accuracy(y_pred_tree)
       

        y_fun = reg_tree.predict([X_train[10]])
        
########################################################################################


#random forest regression
def randForestReg():
        from sklearn.ensemble import RandomForestRegressor
        reg_random_forest = RandomForestRegressor(n_estimators = 10, random_state = 0)
        reg_random_forest.fit(X_train, y_train)

        y_pred_tree_rand = reg_random_forest.predict(X_test)

        accuracy(y_pred_tree_rand)
        return reg_random_forest

def accuracy(y_pred):
        #Accuracy 
        from sklearn import metrics
        from math import sqrt
        rmse = sqrt(metrics.mean_squared_error(y_test,y_pred))
        print('rmse = '+str(rmse))
        print('R2Score = '+str(metrics.r2_score(y_test,y_pred)))    
        
        
def predict(reg):
        
        m = input('Enter Abbreviated month ')        
        d = input('Enter date ')
        y = input('Enter year ')
        
        m = strptime(m,'%b').tm_mon 
        
        x_pred = np.array([m,d,y]).reshape(1,-1)
        
        print('The price will be '+ str(reg.predict(x_pred)).strip('[]'))
         
     
       
decisionTreeReg()
reg = randForestReg()
predict(reg)





