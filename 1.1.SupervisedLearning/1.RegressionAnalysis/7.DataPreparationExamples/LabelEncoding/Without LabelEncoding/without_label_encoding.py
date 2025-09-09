import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('salary_data_transformed.csv')

# encoder = LabelEncoder()
# df['Title'] = encoder.fit_transform(df['Title'])

x = df.drop(['Salary'] , axis =1 )
y = df['Salary']

model = LinearRegression()
model.fit(x , y)

result = model.predict(pd.DataFrame([[1,2]] , columns = ['Title' , 'Experience']))
print(result)
 