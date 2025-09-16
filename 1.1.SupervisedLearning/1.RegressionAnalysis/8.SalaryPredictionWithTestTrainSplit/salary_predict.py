import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
df = pd.read_csv("salary_data.csv")


# plt.scatter(df['Experience'],df['Salary'])
# plt.xlabel('Experience')
# plt.ylabel('Salary')
# plt.title('Experience VS Salary')
# plt.show()

x = df.drop('Salary', axis = 1)
y = df['Salary']

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=1000)

model = LinearRegression()

model.fit(x_train ,  y_train)

train_score = model.score(x_train , y_train)
print(f'the score of the trainng data is {train_score}')

# for exp in x_test.values :
#     y = model.predict([exp])
#     print("the salary of the " + str(exp[0]) + " Years of experiance is " + str(y[0]))


for exp in x_test.values:   
    pred_salary = model.predict([exp])   
    print(f"The salary for {exp[0]} years of experience is {pred_salary[0]}")