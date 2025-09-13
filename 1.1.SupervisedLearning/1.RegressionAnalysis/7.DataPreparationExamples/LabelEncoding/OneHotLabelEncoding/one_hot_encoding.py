import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df =  pd.read_csv("salary_data.csv")

x = df.drop("Salary" , axis =1 )
y = df["Salary"]

column_transformer = ColumnTransformer(
    transformers = [("onehot" , OneHotEncoder(sparse_output= True , drop= "first"), ["Title"])],
    remainder= "passthrough"
)
transformed_values = column_transformer.fit_transform(x)
# print(transformed_values)
# print (column_transformer.get_feature_names_out())

transformerd_feature = pd.DataFrame(transformed_values , columns= column_transformer.get_feature_names_out())

model = LinearRegression()
model.fit(transformerd_feature , y)

data_to_predict = pd.DataFrame( [["Project Manager",8]] ,columns=["Title","Experience"])
# print(data_to_predict)

transformed_col_values = column_transformer.transform(data_to_predict)
# print(transformed_col_values)

transformerd_feature_for_prediction = pd.DataFrame(transformed_col_values , columns=column_transformer.get_feature_names_out())

y_prediction = model.predict(transformerd_feature_for_prediction)


print(f"The salary of the {data_to_predict['Title'][0]} "
      f"with experience of {data_to_predict['Experience'][0]} years "
      f"is: {y_prediction[0]:.2f}")

# print(f"the salary of the {data_to_predict["Title"]} with experice of {data_to_predict["Experience"]} years is: {y}" )