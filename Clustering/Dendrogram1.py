from scipy.cluster import hierarchy
import numpy as np
import matplotlib.pyplot as plt
X=np.array([[0.4005,0.5306],[0.2148,0.3854],[0.3457,0.3156],[0.2652,0.1875],[0.0789,0.4139],[0.4548,0.3022]])
temp = hierarchy.linkage(X, 'single')
dn = hierarchy.dendrogram( temp, above_threshold_color="green", color_threshold=.7)
plt.figure()
plt.show()
