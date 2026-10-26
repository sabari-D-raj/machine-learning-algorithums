import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
import math
from collections import Counter
data=pd.read_csv("knnclassifier/data.csv")
x=data[['Hours','minhour']].values
y=data['Result'].values

trainx,testx,trainy,testy=train_test_split(x,y,test_size=0.2,random_state=42)
k=3
def distance(a,b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
def knn(x):
    distances=[]
    for i in range(len(trainx)):
        d=distance(x,trainx[i])
        distances.append((d,trainy[i]))
        neighbours=distances[:k]
        classes=[item[1] for  item in neighbours]
    return Counter(classes).most_common(1)[0][0]
ypred=[]
for x in testx:
    ypred.append(knn(x))
accuracy=accuracy_score(testy,ypred)
precision=precision_score(testy,ypred,average="weighted",zero_division=0)
recall=recall_score(testy,ypred,average="weighted",zero_division=0)
print("actualclass    predicatedclass")
for i in range(len(testy)):
    print(testy[i],"   ",ypred[i])

print("accuracy",accuracy)
print("precision",precision)
print("recall",recall)


