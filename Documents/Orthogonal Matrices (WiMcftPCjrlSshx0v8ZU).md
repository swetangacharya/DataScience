---
title: Orthogonal Matrices

is_draft: true
---
### Orthogonal Matrices
- **properties of Orthogonal Matrices**  
An orthogonal matrix Q is a square matrix whose rows and columns are orthonormal vectors.

   $$ QQ^T=I $$ or  $Q^T=Q^{-1}$  

- The determinant of an orthogonal matrix is either +1 or -1, i.e., it preserves the volume in Euclidean Space, where +1 indicates rotation and -1 indicates reflection.
- It preserves the inner products of vectors. $u \sdot v = (Q \sdot u) \sdot (Q \sdot v)$, Hence, orthogonal transformations maintain angles and lengths, making them isometries in Euclidean space.
-   They are closed under the multiplication.
- The columns (and rows) of an orthogonal matrix form an orthonormal basis for $ R^n $. This means each column vector has unit length and is orthogonal to the others.
$$ Q = \begin{bmatrix} cos(x) &sin(x) \\ -sin(x) & cos(x) \end{bmatrix}$$ now, 
$$ Q \sdot Q^T$$ is identity matrix.   Also, we can see that length of each row and column is 1.
- The eigenvalues of an orthogonal matrix have a modulus of 1, which means they lie on the unit circle in the complex plane. This characteristic is directly related to the determinant being $\pm1$, reinforcing the notion that orthogonal matrices represent rotations and reflections without scaling.

- **Now, question is when to use orthogonal matrix, i.e., when do we need rotation?** 
   - rotation helps to reduce the correlation between variables. By reducing correlation we can avoid issues that comes with multicolinearity. when two variables are correlated (for ex. number of people in house and water consumption), it makes challenging to interpret the results and draw meaningful conclusions from the model. Less trustworthy statistical conclusions will arise from multicollinearity among independent variables.
   When features are independent or uncorrelated, models can generalize better to unseen data, reducing overfitting and enhancing stability.
   - The original axes may represent correlated variables.
   - After rotation, new axes are aligned with directions that maximize variance while ensuring that these directions (factors) are orthogonal.
  - There are several reasons that we might want to project random vectors onto another basis.  
      1. if we can make correlation between random variables zero, then the
dependence among those variables may be reduced. For instance, if the data is
jointly Normal, then the random variables become independent.
      2.  By projecting onto another basis, the random variables variances also change. For instance, we could project a pair of random variables onto a basis that maximizes the variance in one output random variable and minimizes the variance in the other output random variable.    
##### Symmetric Matrix:-   
It can be written as follows, $A=\Phi \land \Phi^T$, where $\Phi$ is rotational matrix and $\land$ is diagonal Matrix. eigenvectors of $A$ are columns of $\Phi$ (the basis vectors $\hat f_i$),  $\Phi =(\hat f_1|\hat f_2)$, eigenvalue associated with each eigenvector $f_i$ is diagonal elements $\lambda_i$ of $\land$  
$\land=\begin{pmatrix}
   \lambda_1 & 0 \\
   0 & \lambda_2
\end{pmatrix} $



-  #### Dimentionality Reduction ####
   -  We want to transform our data into pieces
of information that are in some sense independent of each other. We will make the data uncorrelated because we do not generally have enough .information to achieve true independence. We want to preserve a set of features that have maximum variance - those where
there are significant variations among the data. Features that have low variance can be well approximated by their mean.  



      **How to reduce correlation?**  
one of the method is to use eigenvalue decomposition of covariance matrix.
say we've highly correlated two variables and their covariance matrix is like below.  
 The eigenvectors of a covariance matrix indicate the directions in which the data varies the most. Let us decompose below covariance matrix. 



$$
\sum=  
 \begin{bmatrix}  
 1 & 0.7 \\
 0.7 & 0.9  
 \end{bmatrix}
$$
![Alt text](correlation1.png)


After decomposition this is what eigenvectors looks like.

$$
eigenvector=  
 \begin{bmatrix}  
 0.73186305 & -0.68145174 \\
 0.68145174 & 0.73186305  
 \end{bmatrix}
$$

now let us project the same data on this new rotated axes in the directions of maximum variance. and we can see that we have decorrelated both variables.
![Alt text](correlation2.png)
$$  
decorelated\ covariance=
\begin{bmatrix}
0.25534126 & 0.09974587 \\
 0.09974587 & 1.64465874
 \end{bmatrix}
$$

### Singular Value Decomposition(SVD)  (geeksforgeeks)
Imagine you have a table of data, like a set of ratings where rows are people, and columns are products. The numbers in the table show how much each person likes each product. SVD helps you split that table into three parts:  
$U$: tells u about the people,like their general preference  
$\sum$: This part shows how important each factor is (how much each rating matters).  
$V^T$: This part tells you about the products (how similar they are to each other) 


### PCA (Principal Component Analysis)  
Now it is easier to understand PCA. Given a multi-dimensional data set, PCA is the process by which the data is decorrelated using the modal matrix of the sample covariance matrix.  
_Then what is modal matrix?_  it is constructed using the eigenvectors of a matrix A as its columns.
- First find the covariance matrix of a standardized data.
- Find the eigenvalue and eigenvector of this covariance matrix.
- now project the standardized data on eigenvectors by multiplication of eigenvector with standardizeddata, like we did in above example.
- One can check the covariance of this new decorelated matrix.  
**_Motivation: principle axis of an Ellipsoid_**  
Consider a d-dimensional normal distribution with mean vector 0 and covariance matrix $\Sigma$. The  pdf is  
$f(x)= \frac{1}{(2\pi|\Sigma|)^{0.5}} e^{\frac{-1}{2}x^T \Sigma^{-1}x}$ where $x \in R^d$  
If we were to draw many iid samples from this pdf, the points would roughly have an ellipsoid pattern, sets of points $x$ such that $x^T \Sigma^{-1} x = c$, for some $c \geq 0$. In particular, consider the ellipsoid where $x^T \Sigma^{−1} x =1$.  
Let $\Sigma=BB^T$, where B is a lower cholesky matrix.**Tthe ellipsoid  can also be viewed as the linear transformation of d-dimensional unit sphere via matrix B**.     
![Alt text](ellipsoid.png)
Moreover, the principal axes of the ellipsoid can be found
via a singular value decomposition (SVD) of B (or $\Sigma$); . In
particular, suppose that an SVD of B is  
$B=UDV^T$ (note: SVD of $\Sigma$ is then $UD^2U^T$)  
The columns of the matrix UD correspond to the principal axes of the ellipsoid, and the relative
magnitudes of the axes are given by the elements of the diagonal matrix D. If some of these
magnitudes are small compared to the others, a reduction in the dimension of the space may be
achieved by projecting each point $x \in R^d$,  onto the subspace spanned by the main $(k\ll d)$, columns of U- so called _principle components._  
All vectors $X$ that satisfies $x^T \Sigma^{-1} x = c$,  lie on the boundary of the same ellipsoid, with size proportional to c. it is a square mahalanobis distance of $X$ from its mean, (assuming mean =0).  
-->
Principal components analysis is sometimes used prior to some factor analytic procedures to determine the  dimensionality of the common factor space. It can also be used to select a subset of variables from a larger set of variables. That is, rather than substituting the principal components for the original variables we can select a set of variables that have high correlations with the principal components. Principal components 
analysis is also used in regression analysis to address multicollinearity problems (i.e., imprecise regression parameter estimates due to highly correlated independent variables).   
Principal components analysis searches for a few uncorrelated linear combinations of the original variables that capture most of the information in the original variables. We construct linear composites routinely, for example, test scores, quality of life indices, and so on. In most of these cases, each variable receives an equal weight in the linear composite. Indices force a p dimensional system into one dimension.   
##### Loadings and Scores:  
In **Principal Component Analysis (PCA)**, **loadings** are coefficients that indicate the contribution of each original variable to the principal components. They essentially show how much each original variable influences the new, derived components. Loadings range from -1 to +1, with larger absolute values signifying a stronger influence of the variable on the component.  `We may say that loadings are understandized eigenvectors' elements. Component loadings are the coordinates of variables onto the components.    
![Alt text](PCA2.png)    
consider only variable V, $a_1,a_2$, are the loadings of V with $F_1$ and $F_2$ respectively. $h'$ is a projection on component plane of vector $h$, which is true position of variable $V$ in a variable space spanned by $V,W,U$. the squared length of the vector is $h^2$, it is the variance of $V$. While $h'^2$ is the portion of that variance explained by two components $F_1$ and $F_2$.   
- $cos\phi$ is a pearson correlation between $V$ and $F_1$. while $cos\alpha$ is a pearson correlation between $h'$ and $F_1$. As a variable $h'$ is a prediction of $V$ by the (standardized) components in linear regression. where loadings $a_1,a_2$ are the regression coefficients (when components are kept orthogonal, as extracted).  
- $a_1=h \cdot cos \phi$, can be understood as scalar product between vector $V$ and unit length vector $F_1$. We take $F_1$ as unit variance vector because it doesn't have its own variance apart from that variance of V which it explains(by amount $h'$).i.e., $F_1$ is extracted from $V,W,U$ and not invited from outside entity. Then clearly, $a_1=(var_v \cdot var_{F_1})^{0.5} \cdot r= h \cdot 1 \cdot cos \phi$, ($var_V=h^2$) and ($var_{F_1}=1$)is the covariance between V and standardized, unit-scaled component $F_1$ take $s_1=(var_{F_1})^{0.5}=1$. This covariance is directly comparable with the covariance between the input variables, for ex. covariance between V and W will be the product of their vector lengths multiplied by cosine between them.  
- To sum up: loading $a_1$  can be seen as the covariance between the standardized component and the observed variable, $h \cdot 1 \cdot cos \phi$, or equivalently between the standardized component and the explained (by all the components defining the plot) image of the variable, $h' \cdot 1 \cdot cos \alpha$. That $cos \alpha$ could be called $V,F_1$ correlation projected on the $F_1,F_2$ component subspace.

**Scores** are the coordinates of your observations (samples) in the new principal component (PC) space created by PCA. Each score indicates an observation's position along a principal component axis. For example, after rotating the axes via PCA, scores show where each sample lies relative to these new axes. The scores matrix has rows for observations and columns for components, summarizing how each observation projects onto each PC.  
 - `In PCA, loadings represent the contribution of each original variable to the new principal components, essentially showing how the original variables are correlated with the components. Scores, on the other hand, represent the position of each data point in the new coordinate system defined by the principal components. Think of loadings as describing the "what" (which variables are important) and scores as describing the "where" (where the data points are located) in the reduced dimension space.` 

- The goal of **PCA** is similar to factor analysis in that both techniques try to explain part of the variation in a set of oserved variables on the basis of a few underlying dimensions.The subtle difference is PCA has no underlying statistical model of the observed variables and it focuses on explainin the total variation in the observed variables on the basis of maximum variance properties of principal components. **Factor analysis**, on the other hand, has an underlying statistical model that partitions the total variance into common and unique variance and focuses on explaining the common variance, rather than the total variance, in the observed variables on the basis of a relatively few underlying fact.
- It is also used in regression analysis to address multicollinearity problems(i.e., imprecise reression parameter estimates due to highly correlated independent variables.)
- **LDA and PCA**:-,while principal components maximize the variance accounted for in the original variables. Linear discriminant function analysis(**LDA**), focusing on differences among groups, determines the weights for a linear composite that maximizes the between group relative to within group variance on that linear composite. **LDA** focuses on maximizing seperability among known categories.
- Principal components analysis searches for a few uncorrelated linear combinations of the original variables that capture most of the information in the original variables.  
- suppose we have p dimensional system mentioning Socio Economic Status (SES),for ex. income, education level, occupational level etc. this can be turned into SES index $y$. so $y=a_1x_1+a_2x_2+...+a_px_p$ 
- in PCA the weights $(a_1,a_2,..,a_p)$ are determined to  to maximize the variation of the linear composite or, equivalently, to maximize the sum of the squared correlations of the principal component with the original variables. The linear composites (principal components) are ordered with respect to their variation so that the first few account for most of the variation present in the original variables, or equivalently, the first few principal components together have, overall, the highest possible squared multiple correlations with each of the original variables.
- **Geomatrically**, the first principal component is the line of closest fit to n observations in the p dimentional variable space. it minimizes the sum of square distances of the n observation from the line in the variable space representing first principal component. equivalently the second principal component is a line closet fit to the residuals from the first principal component. If there are p variables then there is no more than p principal component. There can be fewer if there are linear dependencies among the variables. However there is no advantage in retaining all principal components since we would have as many components as variables and, thus, would not have simplified matters.  
- **Algebraically**, the first principal component, $y_1$, is a linear combination of $x_1,x_2,...,x_p$, (i.e., $y_1=a_{11}x_1+a_{12}x_2+...+a_{1p}x_p = \sum_{i=1}^p a_{1i}x_i$), such that variance of $y_i$ is maximized given the constraint that the sum of squared is equal to 1 $(\sum_{i=1}^p a_{1i}^2=1)$. The random variables, $x_i$, cab be either deviation from mean scores or standardized scores. if the variance $y_1$ is maximized, then so is the sum of squared correlations of $y_1$ with original variables $x_1,x_2,...,x_p$,(i.e., $\sum_{i=1}^p r_{y_1x_i}^2$). PCA finds the optial weight vector ($a_{11},a_{12},...,a_{1p}$,) and associated variance of $y_1$ which is usually denoted by $\lambda_1$.
  
### t-SNE (t-Distributed Stochastic Neighbor Embedding)   
This non-linear, probabilistic method is particularly adept at unraveling the
convoluted topologies of high-dimensional data, making it an invaluable
tool for visualizing and interpreting complex datasets.  
t-SNE operates on the premise of converting the high-dimensional
Euclidean distances between data points into conditional probabilities that
represent similarities. The likelihood that one data point would pick another
as its neighbor is proportional to these probabilities. Subsequently, t-SNE
seeks to minimize the divergence between these probability distributions in
the original high-dimensional space and the low-dimensional embedding.
This is achieved through a gradient descent process on a cost function
known as the Kullback-Leibler divergence.   
t-SNE is particularly effective for datasets where the local structure is of
paramount importance, and it performs admirably in grouping together data
points that are similar. Its ability to reveal clusters at several scales and its sensitivity to local structures make it a preferred choice for exploratory data analysis, especially in the pre-modeling phase.
However, it is essential to recognize the limitations and considerations
associated with t-SNE. The method can be computationally intensive,
particularly for large datasets, and the results can vary depending on the
choice of perplexity parameter and the random seed. Moreover, t-SNE does
not preserve global structures as well as PCA, and the distances in the t-SNE plot do not have a meaningful interpretation.

