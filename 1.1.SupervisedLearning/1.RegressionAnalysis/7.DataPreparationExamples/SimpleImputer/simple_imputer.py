import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

df = pd.read_csv('salary_data.csv')
df.info()

imputer = SimpleImputer(strategy= 'mean')
# imputer = SimpleImputer(strategy= 'median')
# imputer = SimpleImputer(strategy= 'constant' , fit_value = 10000)
# imputer = SimpleImputer(strategy= 'most_frequent')


df['salary'] = imputer.fit_transform(df[['salary']])
df.info()
print (df)