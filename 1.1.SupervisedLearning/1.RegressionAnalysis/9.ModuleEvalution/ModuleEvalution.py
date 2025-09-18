import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score

df = pd.read_csv("SpendingData.csv")
x = df.drop(['Spendings'] , axis=1)
y = df['Spendings']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10000)

plt.scatter(X_train["Salary"], y_train, color='blue', label='Training data')
plt.scatter(X_test["Salary"], y_test, color='red', label='Testing data')
plt.xlabel("Salary")
plt.ylabel("Spendings")
plt.title("Train-Test Data Split")
plt.legend()
# plt.show()


plt.scatter(X_train["Age"], y_train, color='blue', label='Training data')
plt.scatter(X_test["Age"], y_test, color='red', label='Testing data')
plt.xlabel("Age")
plt.ylabel("Spendings")
plt.title("Train-Test Data Split")
plt.legend()
# plt.show()

model = LinearRegression()

model.fit(X_train ,  y_train)

train_score = model.score(X_train , y_train)
test_score = model.score(X_test , y_test)
print(f'the score of the trainng data is {train_score}')
print(f'the score of the test data is {test_score}')

y_pred = model.predict(X_test)
# mean_absolute_error
mae = mean_absolute_error(y_test , y_pred)
print (f'the mean_absolute_error is {mae}')

# mean_squared_error
mse = mean_squared_error(y_test , y_pred)
print (f'mean_squared_error is {mse}')

# root mean squared errror
rmse = np.sqrt(mse)
print (f'root mean squared errror is {rmse}')
