import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("salary_data.csv")
# print(df.columns)
# df.info()
# print(df.describe())

model = LinearRegression()

x = df.drop('Salary', axis = 1)
y = df['Salary']
model.fit(x,y)
# first way to calculate salary
salary = model.predict([[15]])
print(f'Salary of the 15 years of experince is first way {salary}')
# Second way to calculate salary
salary = model.predict(pd.DataFrame([[15]] , columns = ['Experience']))

print(f'Salary of the 15 years of experince is second way {salary}')


# lets calculate here the intercept and coefficient 

coiff = model.coef_
intercept = model.intercept_

# put the vlues in equaton y = mx + c
y = coiff * 15 + intercept
print(f'Salary of the 15 years of experince is calculated from fromula is {y}')

