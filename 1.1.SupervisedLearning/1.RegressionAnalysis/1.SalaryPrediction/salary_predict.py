import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

df = pd.read_csv("salary_data.csv")

print(df.columns)
print (df.describe())

coreletionn = df['Experience'].corr(df['Salary'])
print ("coreletion is: " , coreletionn)

covarience = np.cov  (df['Experience'], df['Salary'])
print ("covrisnce is : " , covarience)


print ("the mean salary of the employee is: " , df['Salary'].mean())
print ("the median salary of the emp is: " ,df['Salary'].median())
print ("the mode salary of the eompoyee is: " , df['Salary'].median())


plt.scatter(df['Experience'],df['Salary'])
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Experience VS Salary')
plt.show()