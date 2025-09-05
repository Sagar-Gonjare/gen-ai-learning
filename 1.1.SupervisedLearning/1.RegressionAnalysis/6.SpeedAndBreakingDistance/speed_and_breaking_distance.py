import pandas as pd 
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt 

df = pd.read_csv("speed_and_breaking_distance.csv")

x = df.drop( columns = ['BrakingDistance'] , axis=1)
y = df['BrakingDistance']

plt.scatter(df['Speed']  , y )
plt.xlabel('Speed')
plt.ylabel('BrakingDistance')
plt.title('Speed vs BrakingDistance')
plt.show()

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(x_poly , y)

output = model.predict(poly.fit_transform([[48]]))
print(output[0])
