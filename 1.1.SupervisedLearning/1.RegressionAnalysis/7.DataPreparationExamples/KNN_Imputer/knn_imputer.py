import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression

df = pd.read_csv('salary_data.csv')
df.info()

imputer = KNNImputer(n_neighbors=5 , )

df['salary'] = imputer.fit_transform(df[['salary']])
df.info()
print (df)