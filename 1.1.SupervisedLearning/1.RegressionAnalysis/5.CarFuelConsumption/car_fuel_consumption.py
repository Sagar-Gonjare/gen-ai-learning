import pandas as pd 
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt 

df = pd.read_csv("car_fuel_consumption.csv")

x = df.drop( columns = ['Fuel_Consumption_LitersPer100km'] , axis=1)
y = df['Fuel_Consumption_LitersPer100km']

plt.scatter(x , y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Engine_Size_Liters vs Fuel_Consumption_LitersPer100km')
plt.show()

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(x_poly , y)

output = model.predict(poly.fit_transform([[0.512]]))
print(output[0])
