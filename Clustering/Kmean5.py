#python data science handbook, 484
from sklearn.metrics import pairwise_distances_argmin
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


def find_clusters(X,n_clusters,rseed=2):
    #1. randomly choose clusters
    rng =np.random.RandomState(rseed)
    i=rng.permutation(X.shape[0])[:n_clusters]
    centers=X[i]

    while True:
        # 2a. assign labels based on closest center
        labels= pairwise_distances_argmin(X,centers)

        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels==i].mean(0) for i in range(n_clusters)])

        # 2c. check for convergence
        if np.all(centers==new_centers): break
        centers = new_centers
    return centers, labels

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);
plt.show()
centers,labels = find_clusters(X,4)
plt.scatter(X[:,0], X[:,1], c= labels, s=50, cmap='viridis')
plt.show()
