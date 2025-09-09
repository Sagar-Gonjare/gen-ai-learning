import pandas as pd 
from sklearn.linear_model import LinearRegression

df = pd.read_csv('salary_data.csv')

train_data = df.dropna()
missing = df[df['salary'].isna()]

x_train = train_data.drop('salary' , axis =1 )
y_train = train_data['salary']

model = LinearRegression()
model.fit(x_train , y_train)

if not missing.empty:
    missing_x = missing.drop('salary' , axis = 1)
    predicted_value = model.predict(missing_x)

    df.loc[df['salary'].isna() , 'salary'] = predicted_value

final_x = df.drop('salary' , axis = 1)
final_y = df['salary']

final_model = LinearRegression()
model.fit(final_x , final_y)

result = model.predict(pd.DataFrame([[4.1]] , columns = ['experience']))
print(result)