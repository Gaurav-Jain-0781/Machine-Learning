# Support Vector Machine
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing the Database
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:].values


# Feature scalling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)


# Creating SVR Model
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(x, y)


# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))


# Visualising Results
plt.plot(x, regressor.predict(x), color="Blue")
plt.scatter(x, y, color="red")
plt.title("Truth Or Bluff")
plt.xlabel("Experience Of Employees")
plt.ylabel("Salary")
plt.show()
