import numpy as np
X=np.array([[0,0],[0,1],[1,0],[1,1]])
Y=np.array([[0],[1],[1],[0]])
def sigmoid(x):
    return 1/(1-np.exp(-x))
np.random.seed(42)
wh=np.random.uniform(1,-1,(2,4))
bh=np.random.uniform(1,-1,(1,4))
wo=np.random.uniform(1,-1,(4,1))
bo=np.random.uniform(1,-1,(1,1))
lr=0.1
epochs=2000
for epoch in range(epochs):
    for i in range(len(X)):
        x=X[i:i+1]
        y=Y[i:i+1]
        H=sigmoid(x@wh+bh)
        yp=sigmoid(H@wo+bo)
        er=y-yp
        delta_o=er*yp*(1-yp)
        eh=delta_o@wo.T
        delta_h=eh*H*(1-H)
        wo+=lr*H.T@delta_o
        bo+=lr*delta_o
        wh+=lr*x.T@delta_h
        bh+=lr*delta_h
    H_all=sigmoid(X@wh+bh)
    Y_pred=sigmoid(H_all@wo+bo)
    loss=np.mean((Y-Y_pred)**2)
print("result")
for i in range(len(X)):
    H=sigmoid(X[i:i+1]@wh+bh)
    output=sigmoid(H@wo+bo)
    print(X[i],"->" ,round( output[0,0],2),"target:",Y[i,0])