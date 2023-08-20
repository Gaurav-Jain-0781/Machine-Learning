#Simple Linear Regression

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing datasets
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[: ,:-1].values
y = dataset.iloc[: , 1].values

# Spliting Data into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=1/3, train_size=2/3, random_state=0)

# Applying Simple Linear Regression to Training Test
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predecting the Test set Results
y_pred = regressor.predict(X_test)

# Visualising the Training Set Results
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary VS Years Of Experience")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test, y_test, color="green")
plt.plot(X_train, regressor.predict(X_train), color="yellow")
plt.title("Salary VS Years Of Experience")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()