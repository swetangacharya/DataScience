import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = load_iris()
df = pd.DataFrame(data.data,columns=data.feature_names)
#print(df.head)

features= ['petal length (cm)', 'petal width (cm)']
x=df.loc[:,features].values
#Apply standardization to features matrix X
x=StandardScaler().fit_transform(x)
y=data.target

#plot
pd.DataFrame(x,columns=features).plot.scatter('petal length (cm)', 'petal width (cm)')
# Add labels
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
#plt.show()
# Make an instance of KMeans with 3 cluster
kmeans=KMeans(n_clusters=3,random_state=1)
# fit only on a features matrix x
kmeans.fit(x)

# Get labels and cluster centroids
labels=kmeans.labels_
centroids=kmeans.cluster_centers_

# visually Evaluate the clusters
x1=pd.DataFrame(x,columns=features)
colormap=np.array(['r','g','b'])
plt.scatter(x1['petal length (cm)'],x1['petal width (cm)'],c=colormap[labels])
plt.scatter(centroids[:,0],centroids[:,1],s=300,marker='x',c='k')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
#plt.show()

# evaluate and compare species
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.scatter(x1['petal length (cm)'],x1['petal width (cm)'],c=colormap[labels])
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('K-Means Clustering (k=3)')

plt.subplot (1,2,2)
plt.scatter(x1['petal length (cm)'],x1['petal width (cm)'],c=colormap[y],s=40)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('Flower Species')
plt.tight_layout()

plt.show()

