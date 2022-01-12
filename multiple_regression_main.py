from matplotlib import pyplot as plt
import numpy as np 
from sklearn import datasets , linear_model
from sklearn.metrics import mean_squared_error as mse
import pandas as pd 

data = pd.read_csv('new_data.csv')

# Data to train the model
train_data_X = data.loc[0:2000,['Profit','Sales','Market_value','Assets']].values
train_data_Y = data.loc[0:2000,['Rank']].values
train_data_X = train_data_X.reshape(2000,4)
train_data_Y = train_data_Y.reshape(2000,1)

# Data to test the model
test_data_X = data.loc[1220:1240,['Profit','Sales','Market_value','Assets']].values
test_data_Y = data.loc[1220:1240,['Rank']].values
test_data_X = test_data_X.reshape(21,4)
test_data_Y = test_data_Y.reshape(21,1)

# Creating a linear regression model using sklearn 
model = linear_model.LinearRegression()
model.fit(train_data_X,train_data_Y)

# Testing data
model_predicted_Y = model.predict(test_data_X)
print(test_data_Y)
print(model_predicted_Y)
error = pow(mse(test_data_Y,model_predicted_Y),1/2)
print(error)

mv = input(" Enter Market Value in Billion $ : ")
assets = input(" Enter Assets Billion $ : ")
sales = input(" Enter Sales Billion $ : ")
profit = input(" Enter profit Billion $ : ")

model_predict = model.predict([[profit,sales,assets,mv]])

print("Your Company's Rank will be from ",model_predict[0][0]+error," to ",model_predict[0][0]-error)


