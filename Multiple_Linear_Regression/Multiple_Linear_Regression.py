# Multiple Linear Regression

# Data Preprocessing

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Datasets
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Categorical Data
# Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Spliting the Data into Test set and Training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, train_size=0.8, random_state=0)

# Applying the Regressor to Training set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicting Test set results
y_predict = lr.predict(X_test)

# Building Optimal Model using Backward Elimination 
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm()
regressor_OLS.endog(y)
regressor_OLS.exog(X_opt)

