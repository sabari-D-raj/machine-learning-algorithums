import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("polynomialregression/data.csv")
print(data)
x=data["Hours studied"].values
y=data["Exam Score"].values

  