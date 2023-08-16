# Data Preprocessing

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Datasets 
datasets = pd.read_csv("Data.csv")
x = datasets.iloc[: ,:-1].values
y = datasets.iloc[: , 3].values
z = datasets.iloc[:, 1:3].values

# Missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
imputer_z = SimpleImputer(strategy='median', missing_values=np.nan)
imputer_z = imputer_z.fit(z[:,:])
z[:,:] = imputer_z.transform(z[:,:])

# Categorial Data
# Independent Variable 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Dependent Variablees
from sklearn.preprocessing import LabelEncoder
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# Spliting the data into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, train_size=0.7, random_state=2)

# Feature Scalling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
z = sc_x.fit_transform(z)

