import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing  import PolynomialFeatures
import matplotlib.pyplot as plt 
from sklearn.pipeline import make_pipeline

df = pd.read_csv("speed_and_breaking_distance.csv")

x = df.drop(columns= ['BrakingDistance'])
y = df['BrakingDistance']

plt.scatter(df['Speed']  , y )
plt.xlabel('Speed')
plt.ylabel('BrakingDistance')
plt.title('Speed vs BrakingDistance')
plt.show()

degree = 2
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

model.fit(x,y)

output = model.predict(pd.DataFrame([[48]]))
print(output[0])