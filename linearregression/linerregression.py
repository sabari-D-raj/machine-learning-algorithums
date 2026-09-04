import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("linearregression/data.csv")
x=data['House size (sq ft)'].values
y=data['Price'].values
#equation pf linear regression is 
#----------y​=b0​+b1​x--------#
#x → input/feature
#ŷ → predicted output
#b₀ → intercept
#b₁ → slope/weight
x_mean=sum(x)/len(x)
y_mean=sum(y)/len(y)
numerator=sum((xi-x_mean)*(yi-y_mean)for xi,yi in zip(x,y))
denominator=sum((xi-x_mean)**2 for xi in x)
slopeb1=numerator/denominator
interceptb0=y_mean-slopeb1*x_mean
Y_pred=slopeb1*x+interceptb0
house_size=int(input("enter the house size:"))
predicted_price=slopeb1*house_size+interceptb0
print("------house price----")
print(f"{predicted_price:.2f} lakhs")
error=y-Y_pred
mse=sum(error**2)/len(y)
mae=abs(error)/len(y)
print("-----error-----")
print(f"mse={mse:.2f}")
print(f"mae={mae}")
plt.scatter(x,y)
plt.plot(x,Y_pred)
plt.xlabel("house price")
plt.ylabel("price")
plt.show()