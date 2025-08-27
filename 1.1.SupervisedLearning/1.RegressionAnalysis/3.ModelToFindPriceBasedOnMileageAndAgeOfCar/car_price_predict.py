import numpy as np 
import pandas as pd
from sklearn.linear_model  import LinearRegression
import matplotlib.pyplot as plt 


df = pd.read_csv("car_price_data.csv")

df.info()

corr_age = df['Age'].corr(df['Price'])
print (f'coreletion between Age and price is {corr_age}')

corr_mil = df['Mileage'].corr(df['Price'])
print (f'coreletion between Mileage and prize is {corr_mil}')

x = df.drop(['Price'] , axis = 1)
# print (x)
y = df['Price']

model = LinearRegression()

model.fit(x,y)

pred_first_way = model.predict([[1,30000]])
print (f'the prize of the car which is used for 1 year and  30000 km runnig is {pred_first_way}')

prediction = model.predict(pd.DataFrame([[2,20000],[3,120000]] , columns = ['Age' , 'Mileage']))
print (f'the prize of the car which is used for 2 year and  20000 km runnig is {prediction[0]} which is zero th index')

print (f'the prize of the car which is used for 3 year and  120000 km runnig is {prediction[1]} which is zero th index')

