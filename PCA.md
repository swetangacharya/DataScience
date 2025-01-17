# **PCA  (Principal Component Analysis)**  
-  it is a method to linearly transform a domain into a co-domain whose basis vectors are orthogonal.  
This transformation is often considered for dimensionality reduction via feature selection, for studying the
data variance along reference axes, as well as for analyzing data subspaces.  
- Like SVD, we will order them by successive importance, so we can reconstruct an approximate representation using few components. PCA and SVD are so closely related as to be indistinguishable for our purposes.   
They do the same thing in the same way, but coming from different directions.
  <h2> When should we use PCA? </h2>  
-  when we have large set of correlated variables, principal components allow us to summarize this set with a smaller number of representative variables that collectively explain most of the variability in the original set.  
  It is an unsupervised learning as it involves the set of features without its labels.  
  Now suppose we wish to visualize n observation with measurement on a set of p features, X<sub>1</sub>,X<sub>2</sub>,...,X<sub>n</sub>, as part of an exploratory data analysis. 
  -  We could do this by examining two-dimensional scatterplots of the data, each of which contains the n observations’ measurements on two of the features.But the problem is, for p=10, we need 45 scatter plots, hence with large p, it would be hard to look at all of them. also none of them will be informative because they each contains only small fraction of information the of the total dataset.  
  -  Then what we need is a low-dimensional representation of the data that captures as much of the information as possible. PCA does it exactly.
  -  **PCA** finds a low-dimensional representation of a data set that contains as much as possible of the variation. The
idea is that each of the n observations lives in p-dimensional space, but not all of these dimensions are equally interesting. PCA seeks a small numberof dimensions that are as interesting as possible, where the concept of interesting is measured by the amount that the observations vary along each dimension. Each of the dimensions found by PCA is a linear combination
of the p features. We now explain the manner in which these dimensions,
or principal components, are found

$$  z_{i1}=\phi_{11}x_{i1}+\phi_{21}x_{i2}+...+\phi_{p1}x_{ip} $$




The (&phi; <sub>11</sub>,&phi; <sub>21</sub>,...,&phi; <sub>p1</sub>) are the loadings of first principal components PC1.
and (x <sub>i1 </sub>, x <sub>i2 </sub>,..., x<sub>ip </sub>) are the measured values of respective variables for a person1. i.e., The standardised (mean=0, std=1) data for each person.  
- z <sub>11 </sub> is the new value for the person1 with axis of PC1. and z_{21} is the new value for person2 with axis PC1.
- The old variables ( Diastolic BP, Systolic BP, Weight, Hight) is replaced by (PC1,PC2,PC3,PC4).
- We refer $z_{11},z_{21},...z_{n1}$ as scores of first principal component PC1 and $z_{12},z_{22},...z_{n2}$ as scores of second principle component PC2, and so on.
- so what we are trying to achieve is  
$$maximize_{(\phi_{11},\phi_{21},...,\phi_{p1)}}{\Sigma_{i=1} ^n (\Sigma_{j=1}^p (\phi_{j1}x_{ij}))^2}$$



