import pandas as pd 
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt 

df = pd.read_csv("data.csv")

x = df.drop( columns = ['b'] , axis=1)
y = df['b']

plt.scatter(x , y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('a vs b')
plt.show()

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(x_poly , y)

output = model.predict(poly.fit_transform([[512]]))
print(output[0])
