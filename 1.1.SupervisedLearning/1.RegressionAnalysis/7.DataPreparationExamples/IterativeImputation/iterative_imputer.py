import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression

df = pd.read_csv('salary_data.csv')
df.info()

imputer = IterativeImputer(estimator=BayesianRidge(), random_state=0, max_iter=5  )

df['salary'] = imputer.fit_transform(df[['salary']])
df.info()
print (df)