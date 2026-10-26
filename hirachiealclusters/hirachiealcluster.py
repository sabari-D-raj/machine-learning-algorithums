import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster,linkage,dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
x_scal=StandardScaler().fit_transform(x)
z=linkage(x_scal,method='ward')
plt.figure(figsize=(6,6))
dendrogram(z)
plt.title("dendrogram")
plt.xlabel("sampleindex")
plt.ylabel("Euclediean distance")
plt.show()
k=3
cluster=fcluster(z,k,criterion='maxclust')
agg=AgglomerativeClustering(n_clusters=k,linkage='ward')
label=agg.fit_predict(x_scal)
plt.scatter(x_scal[:,0],x_scal[:,1],c=label,cmap='rainbow')