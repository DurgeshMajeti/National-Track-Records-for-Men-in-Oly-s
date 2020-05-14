from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from main import data

data = data.T
X=np.array(data[:,:8]).reshape(55,8)

#Find number of K
distrotions = []
K = range(1,11)
for k in K:
    kmeanModel = KMeans(n_clusters = k).fit(X)
    kmeanModel.fit(X)
    distrotions.append(sum(np.min(cdist(X,kmeanModel.cluster_centers_,'euclidean'),axis = 1))/X.shape[0])

plt.plot(K,distrotions,'bx-')
plt.xlabel('k')
plt.ylabel('Distortions')
plt.show()

#ploting cluster for k = 4
plt.show()
km = KMeans(n_clusters = 4,init = 'random',n_init = 10,max_iter=300, tol = 1e-04,random_state = 0)
y_km = km.fit_predict(X)
plt.scatter(X[y_km == 0,0],X[y_km == 0,1],s = 50, c='red',edgecolor= 'black',label = 'cluster1')
plt.scatter(X[y_km == 1,0],X[y_km == 1,1],s = 50, c='green',edgecolor= 'black',label = 'cluster2')
plt.scatter(X[y_km == 2,0],X[y_km == 2,1],s = 50, c='blue',edgecolor= 'black',label = 'cluster3')
plt.scatter(X[y_km == 3,0],X[y_km == 3,1],s = 50, c='lightgreen',edgecolor= 'black',label = 'cluster4')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s = 250,marker='*',c = 'black',edgecolor='black',label='centroids')
plt.show()
clusters = {}
km.cluster_centers_ -= km.cluster_centers_%0.001
for i in range(len(km.cluster_centers_)):
    clusters[tuple(km.cluster_centers_[i])] = X[y_km == i]
for i in range(len(clusters)):
    print(i+1,") ",km.cluster_centers_[i]," :\n",clusters[tuple(km.cluster_centers_[i])])
    print()