from matplotlib import pyplot as plt
import numpy as np 
from sklearn import datasets , linear_model
from sklearn.metrics import mean_squared_error as mse
import pandas as pd 

data = pd.read_csv('new_data.csv')

# Data to train the model
train_data_X = data.iloc[0:20,6]
train_data_Y = data.iloc[0:20,0]

# Data to test the model
test_data_X = data.iloc[0:200,6]
test_data_Y = data.iloc[0:200,0]

m=0
c=0
n=len(train_data_X)
LR = 0.001

for i in range(n):
    model_predicted_Y = m*train_data_X+c
    D_m = (2/n) * sum(train_data_X * (train_data_Y-model_predicted_Y))
    D_c = (2/n) * sum(train_data_Y-model_predicted_Y)

    m = m - LR * D_m
    c = c - LR * D_c

print(" m = ",m," c = ",c)
Y_predict = m * test_data_X + c
print(Y_predict)
# Visualizing Results
plt.xlabel("Company's Profit")
plt.ylabel("Company's Rank")
plt.scatter(test_data_X,test_data_Y)
plt.plot(test_data_X,Y_predict)
plt.show()

print(pow(mse(test_data_X,Y_predict),1/2))