import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("polynomialregression/data.csv")

x=data["Hours studied"].values
y=data["Exam Score"].values

M=np.c_[np.ones((len(x),1)),x,x**2]
beta=np.linalg.inv(M.T@M)@M.T@y
y_pred=M@beta
h=int(input("enter hour"))
hM=np.array([1,h,h**2])
p=hM@beta
print(p)
plt.scatter(x,y)
plt.plot(x,y_pred)
plt.xlabel("hours studied")
plt.ylabel("Exam score")
plt.show()