import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
data=pd.read_csv("linearregression/data.csv")
x=data['House size (sq ft)'].values.reshape(-1,1)
y=data['Price'].values
model=LinearRegression()
model.fit(x,y)
house_price=int(input("enter the house size"))
y_pred=model.predict(x)
house_prices=model.predict([[house_price]])
print("house price",house_prices)
mse=mean_squared_error(y,y_pred)
mae=mean_absolute_error(y,y_pred)
print(mse)
print(mae)
