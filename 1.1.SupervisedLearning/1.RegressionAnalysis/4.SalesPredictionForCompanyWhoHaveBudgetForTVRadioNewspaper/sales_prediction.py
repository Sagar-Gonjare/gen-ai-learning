import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
df= pd.read_csv("Advertising.csv")

df.info()

corr_tv = df['TV'].corr(df['sales'])
print (f'coreleetion ratio between TV And sales {corr_tv}')

corr_radio = df['radio'].corr(df['sales'])
print (f'coreleetion ratio between radio And sales {corr_radio}')

corr_newspaper = df['newspaper'].corr(df['sales'])
print (f'coreleetion ratio between newspaper And sales {corr_newspaper}')

# plt.scatter(df['TV'],df['sales'])
# plt.xlabel('TV')
# plt.ylabel('sales')
# plt.title('TV VS sales')
# plt.show()

# plt.scatter(df['radio'],df['sales'])
# plt.xlabel('radio')
# plt.ylabel('sales')
# plt.title('radio VS sales')
# plt.show()

# plt.scatter(df['newspaper'],df['sales'])
# plt.xlabel('newspaper')
# plt.ylabel('sales')
# plt.title('newspaper VS sales')
# plt.show()

x = df[['TV' , 'radio' , 'newspaper']]
y = df['sales']

model = LinearRegression()

model.fit(x,y)

sales = model.predict([[230.1,37.8,69.2]])
print(f'sales is {sales[0]}')

sales_2 = model.predict(pd.DataFrame([[23.1,370.8,69.2]] , columns = ['TV' , 'radio' , 'newspaper']))
print(f'sales is {sales_2[0]}')