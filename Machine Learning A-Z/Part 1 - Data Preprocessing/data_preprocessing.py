#data_preprocessing.py
import numpy as nu
import matplotlib.pyplot as plt 
import pandas as pd 

#Set the Data.csv folder as the current directory
dataset = pd.read_csv('Data.csv')
#independent variables
#fetch [All the lines, all the columns except last column]
x = dataset.iloc[:, :-1].values
#dependent variables ie the last column- Purchase 
y = dataset.iloc[:, 3].values

#Handing Missing Data: Use Mean of the remaining Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#Fit the mean result into the dataset where the value is NaN 
imputer = imputer.fit(x[:, 1:3])
#replace the missing data 
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encoding categorical data
# Country and Purchase are categorical data 
# Replace text with numbers since ml is mathematical model, good to replace text with number 
# Dummy encoder: To make sure that the ml model will not consider the order of the numerical values
# create one column for each categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Split data into Training set and Test set 
from sklearn.cross_validation import train_test_split
# 20% data will be used for training maximum 40%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# ML Model is based on Euclidean Distance 
# Euclidean Distance between P1 and P2 = Sqrt((x2-x1)2 + (y2-y1)2)
#Standardisation x(std) = (x - mean(x))/SD(x)
#Normalization x(norm) = (x - min(x))/(max(x) - min(x))

# Feature Scaling
# Convert data into same range to avoid the impact of large data in one column
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
#Train data already fix and so test data not needed to fit againg to the model.
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)


