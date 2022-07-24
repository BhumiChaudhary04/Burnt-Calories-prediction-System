import pickle
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Loading the dataset from calories.csv file to Pandas DataFrame
calories = pd.read_csv('calories.csv')

# First 5 rows of the dataframe
calories.head()

# Loading the dataset from exercise.csv file to Pandas DataFrame
exercise_data = pd.read_csv('exercise.csv')

# First 5 rows of the dataframe
exercise_data.head()

calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

# number of rows and columns of dataset
calories_data.shape

# information about the dataset
calories_data.info()

# checking for missing values
calories_data.isnull().sum()

# get some statistical measures about the data
calories_data.describe()

sns.set()

# plotting the "gender" column in count plot
sns.countplot(calories_data['Gender'])

sns.distplot(calories_data['Age'])

sns.distplot(calories_data['Height'])

sns.distplot(calories_data['Weight'])

correlation = calories_data.corr()

# heatmap to understand the correlation

plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Greens')


calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)

calories_data.head()

X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calories_data['Calories']

print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

model = XGBRegressor()

model.fit(X_train, Y_train)

test_data_prediction = model.predict(X_test)
print(test_data_prediction)

mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("Mean Absolute Error = ", mae)



pickle.dump(model, open("ml.sav", "wb"))




