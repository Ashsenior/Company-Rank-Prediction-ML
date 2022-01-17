import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

# Reading data from csv file
data = pd.read_csv('new_data.csv')

# Dividing data into Labels (Y) and features (X)
X = data.drop(['Name','Rank'],axis=1)
Y = data['Rank']

# Transforming Categorical feature 'Country' into numerical value
X_transformed = pd.get_dummies(X)

# Dividing data into train and test sets
x_train,x_test,y_train,y_test = train_test_split(X_transformed,Y,test_size=0.2)

# Creating RandomForestRegressor model
model = RandomForestRegressor()
# Training the model with the train set
model.fit(x_train,y_train)

# Predicting the Labels 'Rank' with test data
y_pred = model.predict(x_test)

# Using MATPLOTLIB to analyse the results 
plt.style.use('seaborn-darkgrid')
fig, ax = plt.subplots(figsize=(5,5))
ax.set(title='Prediction Result',xlabel='Predicted Rank',ylabel='Actual Rank')
ax.scatter(y_pred,y_test,c='green')
plt.show()

# Printing the SCORE / ACCURACY of the model with test data
print(" Model score is : ",model.score(x_test,y_test)*100,"%")