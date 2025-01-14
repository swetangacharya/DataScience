# **PCA  (Principal Component Analysis)**  
-  it is a method to linearly transform a domain into a co-domain whose basis vectors are orthogonal.  
This transformation is often considered for dimensionality reduction via feature selection, for studying the
data variance along reference axes, as well as for analyzing data subspaces.  
- Like SVD, we will order them by successive importance, so we can reconstruct an approximate representation using few components. PCA and SVD are so closely related as to be indistinguishable for our purposes.   
They do the same thing in the same way, but coming from different directions.
  <h2> When should we use PCA? </h2>  
  when we have large set of correlated variables, principal components allow us to summarize this set with a smaller number of representative variables that collectively explain most of the variability in the original set.  
  It is an unsupervised learning as it involves the set of features without its labels.  
  Now suppose we wish to visualize n observation with measurement on a set of p features, X<sub>1</sub>,X<sub>2</sub>,...,X<sub>n</sub>

