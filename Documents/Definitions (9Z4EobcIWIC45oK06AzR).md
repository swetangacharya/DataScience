---
title: Definitions
description: Shows all important definitions.
is_draft: true
tags:
  - Linear Algebra
  - Statistics
---

:::info
**New here?**  
One can find definition and short explanation of Statistics and Data Science.


---
::::
#### A  
[1] **Asymptotical:-**  refers to the behavior of a function, sequence, or process as it approaches a limiting value, typically as some parameter (often size or time) goes to infinity. In mathematics, if a function is said to have asymptotic behavior, it means that as the input grows very large, the function gets arbitrarily close to a certain value or follows a certain trend, even if it never actually reaches it.

[2] **Accuracy:-** it is a number of correctly classified examples divided by total number of classified examples. Accuracy = $\frac{TP+TN}{TP+TN+FN+FP}$

[3] **Associative Mapping:-**  In neural networks, associative mapping refers to the process of learning and recalling relationships between input and output patterns, allowing the network to retrieve an associated output pattern when presented with a partial or noisy version of a related input.

#### B  
**[1] Binomial Distribution**
  * it is discrete, gives the probability of getting certain number of success after repeating same experiment multiple times, independently. Here, n=number of times we perform experiment, p= predefined probability of success.   
  Real-world examples include predicting the number of patients that will develop side effects for a vaccine or a new medication in a clinical trial, the number of ad clicks that will result in a purchase, and the number of customers that will default on their monthly credit card payments.  
  When we model examples from the real world using a probability distribution that requires independent trials, it means that we are assuming independence even if the real-world trials are not really independent. It is good etiquette to point our models’ assumptions.    
[2] **Bias:-** Bias generally measures the average difference between the predicted values and the true values. It indicates whether your predictions systematically over- or under-estimate the true values. for example if we have Y = [5, 8, 12, 10, 15],Y_pred = [4, 7, 10, 11, 13], then  Bias = $ \color{darkorange}\frac{1}{n}\sum_{i=1}^n(Y_{pred,i}-Y_i)=-1$  , this means our model is under-predicting by 1 unit.
 in this case. now Errors $\color{darkorange}e_i= Y_{pred,i}-Y_i = [-1,-1,-2,1,-2]$,  
Now, **Variance** = $\frac{1}{n}\sum_{i=1}^n(e_i- Bias)^2=1.2$, so **variance** here is best interpreted as the variance of the prediction errors relative to their average prediction (bias removed).  
**Bounded Set:-** A set of real numbers is said to be bounded if it is bounded above as well as below. When set S is bounded, $\exists$ two real numbers a and b, such that  
  $\color{green}a\le x \le b, \forall \ x\in S$
  
#### C
  **[1] Covariate**  
    In a linear model $y=\beta_0+\beta_1x_1+...+\beta_kx_k+\epsilon$, the $x_1,x_2,x_k$ are called covariates, or regressor variables.  
    **[2] confounding variable**  
    A variable that is related both to group membership and to the response variable. Two variables are confounded if their effects on the response variable cannot be distinguished from one another.  
  **[3] Chebychev's Rule:**:  
   consider any number k, (k>=1), then percentage(%) of any observation within k standard deviation of the mean is at least $100(1-\frac{1}{k^2})$%  .    
   - Let $ X$ be a R.V having finite mean $\mu$ and finite variance $\sigma^2$, let $k \in \R^+$, then Chebyshev's inequality holds:  
$\color{brown}P(|X-\mu|\ge k)\le \frac{\sigma^2}{k^2}$  
- Say $k= n\sigma$, then $\color{brown}P(|X-\mu|\ge n\sigma)\le \frac{1}{n^2}$, i.e., 
   - a minimum of just 75% of values must lie within two standard deviations of the mean and 88.88% within three standard deviations for a broad range of different probability distributions.   i.e. probability of R.V X away from 5 std. dev. is $ \le\frac{1}{n^2}$, i.e., $\le \frac{1}{25}$  
   - $P(|X-\mu|\ge k\sigma)\le \frac{1}{k^2}$   
  $P(|X-\mu|< k\sigma)\ge 1-\frac{1}{k^2}$  
  $P(\mu-k\sigma< X < \mu+k \sigma)\ge 1-\frac{1}{k^2}$    
  $P(|X-\mu|\le \epsilon)> 1-\frac{\sigma^2}{\epsilon^2}$
    
   **[4] Chi-Squared Distribution:-**  
   This is continuous. Special case of Gamma Distribution where $\lambda=1/2$ and $\eta=n/2$ , it is a distribution for the sum of squares of normally distributed independent random variables. You might wonder why would we care about squaring normally distributed random variables, then adding them up. The answer is that this is how we usually compute the variance of a random variable or of a data sample, and one of our main goals is controlling the variance in order to lower our uncertainties. There are two types of significance tests associated with this distribution: the goodness of fit test, which measures how far off our expectation is from our observation, and the independence and homogeneity of data features test.  

  **[5] Correlation coefficient**:- is the sum of products of the standardized variables divided by n-1.  The magnitude of r describes the strength of a linear relation and its sign indicates the direction.
$r= \frac{1}{n-1} \sum_{i=1}^n (\frac{x_i-\bar x}{s_x})(\frac{y_i-\bar y}{s_y})$ also can be written as $r=\frac{S_{xy}}{(S_{xx} \cdot S_{yy})^{0.5}}$  
r=0, implies that the linear association between two variables is weak  
r=+1,implies that all pairs $(x_i,y_i)$ lie exactly on straight line having a positive slope.
r=-1,implies that all pairs $(x_i,y_i)$ lie exactly on straight line having a negative slope.    
Now, **The proportion of the y variability explained by linear relation** is   
(Sum of squares due to regression)/(Total sum of squares of y)
= $\frac{S^2_{xy}}{S_{xx}S_{yy}}= r^2$    
Correlation coefficient measures linear association.  
**[6]Critical Point:-**  An iterior point of the domain of a function $f$ where $f'$ is 0 or undefined is a critical point of $f$.  
**[7]Curvature of a Plane Curve:-**  
As a partical moves along a smooth curve in the plane, $T=dr/ds$, turns as the curve bends. $T$ is a unit vector,hence only direction changes as partical moves along the curve. The rate at T turns per unit lengthalong the curve is called the curvature. see figure below.  
![Alt text](curvature.png)  
if $T$ is a unit tangent vector of a smooth curve in the plane then the **curvature** function of curve is $\kappa= |dT/ds|$

**[8]Conditional Notation:-**  
In plain terms, **$p(x|z; \theta)$,** represents the probability of observing $x$ given that $z$ has occured, under the model parameterized by $\theta$. The semicolon (;) before $\theta$  is often used to distinguish parameters from random variables—parameters are treated as fixed for the purpose of the probability statement.  
 while **$p(x|z, \theta)$,**  which sometimes (depending on context) is used to indicate joint conditioning on $z$ and $\theta$.    
**[9] Cramer-Rao Lower Bound:-** States that for any unbiased estimator $\hat \theta$ of a parameter $\theta$, the variance of $\hat \theta$ is bounded from below by inverse of Fisher Information.
$\color{darkblue}Var(\hat \theta) \ge \frac{1}{I(\hat \theta)}$  
**[10] Conditional Probability:-**  
$p(A=a|B=b)$ is the probability of A being in a state under the constraint that B is in state b'.  
**[11] Characteristic value and Characteristic vector:-**  In linear algebra, characteristic values (also called eigenvalues) and characteristic vectors (also called eigenvectors) are fundamental concepts that describe how a linear transformation (represented by a matrix) acts on vectors. Eigenvalues are scalar values that indicate how much an eigenvector is scaled by the transformation, while eigenvectors are non-zero vectors whose direction remains unchanged (or is reversed) by the transformation.  
**[12] Covariance:-**  covariance measures the joint variability of the deviations from the means, and adding a constant shifts the mean but does not affect the variability or the covariance. hence $Cov(X+a,Y+b)= Cov(X,Y)$

**[13] Clustering:-**  Partition of Dataset D into k subsets is called clustering, $C_1,C_2...C_k$.   
$\color{magenta}\bigcup_{i=1}^k C_i=D $  and $\color{magenta}\bigcap_{i=1}^k C_i= \phi$.  
A “good” cluster is a subset of
points which are closer to the **mean** of their own cluster than to the **mean** of other clusters.

#### D  
**[1] directional derivative:**  
It describes the rate at which a function changes at a specific point in a specified direction, specifically given a unit vector $u$ and an arbitrary point $(x_0,y_0)$ , the directional derivative tells us how fast the function $f(x,y)$ is changing at $(x_0,y_0)$ in the direction of $u$. We know that dot product between two vector tells us how two vectors go togather.  
$\Delta f(x_0,y_0)\cdot u=||\Delta f(x_0,y_0)||||u||cos(\theta) $
if directional vector length $||u||=1$ then  
$\Delta_u f(x_0,y_0)= |\Delta f(x_0,y_0)|| \ ||u|| \  cos(\theta)$   

-> $df=f'(P_0)ds$ , Ordinary derivative x increment, for two or more variables,  
$df = (\Delta f|_{P_0} \cdot u) ds$ , directional derivative x increment

  * Examples: estimate how much the value of $f(x,y,z)=ysinx+2yz$ will change if point $P(x,y,z)$ moves 0.1 unit from $P_0(0,1,0)$ straight toward $P_1(2,2,-2)$.  
  now $\overline {P_0P_1}=2i+j-2k$, The direction of this vector is   
  $u= \frac{\overline {P_0P_1}}{|P_0P_1|}= \frac{\overline {P_0P_1}}{3}= \frac{2}{3}i+\frac{1}{3}j-\frac{2}{3}k$   
  $\Delta f|_{(0,1,0)}=((ycosx)i+(sinx+2z)j+2yk|_{(0,1,0)}=i+2k$  
  Therefore, $\Delta f|_{P_0} \cdot u =(i+2k) \cdot (\frac{2}{3}i+\frac{1}{3}j-\frac{2}{3}k) = -\frac{2}{3} $  
  The change df in f that results from moving $ds=0.1$ units away from $P_0$ in the direction of $u$ is approximately,  
  $df = \Delta f|_{P_0} \cdot u = (-\frac{2}{3}) (0.1) =-0.067$ units.

**[2] Gradient:-** The gradient of a two-variable function $f(x, y)$ is a vector, denoted by $\nabla(x, y)$, whose components are the partial derivatives of $f(x, y)$:  
$\color{orange}\nabla f(x, y)=  \dbinom{f_x(x,y)}{f_y(x,y)}$  
  - **properties of gradient**: Let $f(x, y)$ be a smooth two-variable function, let the input $(a, b)$ be a point on the plane, and let the vector $\nabla(a, b)$ be graphed at $(a, b)$ on the contour plot of $f(x, y)$  
    1.  Vector $\nabla f(a, b)$ points in the direction of greatest increase of $f(x, y)$ at $(a, b)$,  

    2.  -$\nabla f(a, b)$points in the direction of greatest decrease of $f(x, y)$ at $(a, b)$   
    3. the length of $\nabla f(a,b)$ measures the steepness of $f(x, y)$ at $(a, b)$.  
    4. At $(a,b)$ , $\nabla f(a,b)$is perpendicular to the contour of f(x, y) at level f(a, b).
### Diagonalisation of a Matrix (Similarity Tranformation):-  
Let A and B be two square matrices of order n. Then B is said to be a similar to A, if there exists a non-singular matrix P such that  
   - $\color{orange}B=P^{-1} A P$,  it is called similar transformation.  

Diagonalisation of a matrix A is the process of reducing the matrix A into a diagonal form D. if matrix A is related to D by a similarity transformation such that $D=P^{-1} A P$, then  the matrix A is reduced ot the diagonal matrix D through modal matrix P.   
  - Note (a): Matrix D is called spectral matrix of A.  
     (b) if a suqare matrix A of order n has n linearly independenteigen vectors, then a matrix P can be found such that $\color{orange}B=P^{-1} A P$, is a diagonal matrix.  
     (c) The square matrix which diagonalises A, is found by grouping eigen vectors of A into a square matrix and the resulting diagonal matrix has the eigenvalues of A as its diagonal elements.  
     (d) the reduction of A ot a diagonal matrix is, obviously a particular case of similarity transformation.  
     (e) The matrix P which diagonalises A is called the modal matrix of A and the resulting diagonal matrix D is known as spectra matrix of A. 
     Example:  
     Let A = $\begin{bmatrix}
          -2 & 2 & -3\\
          2 & 1 & -6\\
          -1 & -2 & 0
          \end{bmatrix}$  
    ```python
    import numpy as np
    A = np.array([[-2, 2,-3], [2, 1,-6],[-1,-2,0]])
    eigenvalues, eigenvectors = np.linalg.eig(A)  
    D = np.diag(eigenvalues)
    P=eigenvectors
    P_inv = np.linalg.inv(P)
    reconstructed_A = P @ D @ P_inv  # that is same as diagonal eigenvalues.
     new_D= np.round(P_inv@A@P, decimals=2)
    ```


  eigenvalues = array([-3.,  5., -3.])  

  eigenvectors=    
  ([[-0.95257934,  0.40824829, -0.02296692],  
       [ 0.27216553,  0.81649658,  0.83534731],  
       [-0.13608276, -0.40824829,  0.54924256]])
  
  reconstructed_A= 
        ([[-2.,  2., -3.],
       [ 2.,  1., -6.],
       [-1., -2., -0.]])  
    new_D = 
    ([[-3.,  0.,  0.],
       [ 0.,  5., -0.],
       [ 0.,  0., -3.]])

  
#### E  
* **extraneous variable**  
An extraneous variable is one that is not one of the explanatory variables in the study but is thought to affect the response variable .  
* **Empirical Rule:**  
![Alt text](Empirical_normal.png)  

If the histogram of values in a data set can be reasonably well approximated by a normal
curve, then 
Approximately 68% of the observations are within 1 standard deviation of the mean.
Approximately 95% of the observations are within 2 standard deviations of the mean.
Approximately 99.7% of the observations are within 3 standard deviations of the mean.    

* **Exponential Distribution:-**  
This is continuous. If we happen to know that a certain event occurs at a constant
rate λ, then exponential distribution predicts the waiting time until this event
occurs. It is memoryless, in the sense that the remaining lifetime of an item
that belongs to this exponential distribution is also exponential. The controlling
parameter is the constant rate λ. Real-world examples include the amount of
time we have to wait until an earthquake occurs, the time until someone defaults
on a loan, the time until a machine part fails, or the time before a terrorist attack
strikes. This is very useful for the reliability field, where the reliability of a certain
machine part is calculated, hence statements such as a 10-year guarantee, etc.  

**Evidence Lower Bound:- (ELBO)**
The evidence lower bound (ELBO) is an important quantity that lies at the core of a number of important algorithms in probabilistic inference such as expectation-maximization and variational infererence. To understand these algorithms, it is helpful to understand the ELBO.
#### F  
**F1 Score:-** it is a harmonic mean of **precision and recall.**  
$F_1= \frac{2}{\frac{1}{recall} X\frac{1}{precision}}$  
$F_1= 2 (\frac{precision \cdot recall}{precision+recall})$  
$F_1=\frac{TP}{TP+\frac{1}{2}(FP+FN)}$

**Feature Selection and Feature Extraction:-**  
**Feature Selection:-** involves selecting a subset of the
most relevant features from the original dataset. The premise here is that not all features contribute equally to the predictive power of a model. By
eliminating redundant or irrelevant features, we can enhance model
performance by reducing overfitting, improving accuracy, and sometimes
even reducing training time.    
Feature selection refers to a process whereby a data space is transformed
into a feature space that, in theory, has exactly the same dimension as the original data space.  
However, the transformation is designed in such a way that the data set may be
represented by a reduced number of “effective” features, yet retain most of the intrinsic information content of the original data; in other words, the data set undergoes a
dimensionality reduction. To be specific, suppose we have an m-dimensional vector $x$ and wish to transmit it using $l$ numbers, $l\lt m$ , which implies that data compression is an intrinsic part of feature mapping. if we simply truncate the vector $X$ , we will cause mean square error equal to the sum of the variances of the elements eliminated from x, so we ask the following question ...  
_Does there exist an invertible linear transformation $T$ such that the truncation of $Tx$ is optimum in the mean-square-error sense?_  
Clearly, the transformation T should have the property that some of its components have low variance.

**Feature Extraction:-**  is about transforming the data into a new
space of features. Rather than choosing from existing features, it creates
new ones by combining the information from the original set in various
ways. This can lead to a reduction in dimensionality, similar to what we
observed with autoencoders, but it does so by constructing new variables
that capture the essence of the data in a more efficient form.    
**Fuzzy Set:-** Fuzzy logic starts with the concept of a fuzzy set. A fuzzy set is a set without a crisp, clearly defined boundary. It can contain elements with only a partial degree of membership.  
The only condition a membership function must satisfy is that its membership values must vary between 0 and 1. The function itself can be an arbitrary optimized for your desired combination of simplicity, convenience, speed, and efficiency.
A classical set might be expressed as:  
$\color{blue}A=\{x|x>6\}$  
A fuzzy set is an extension of a classical set. if $X$ is the universe of discourse and its elements are denoted by $x$, then a fuzzy set $A$ in $X$ is defined as a set of ordered pairs  
$\color{blue}A=\{x,\mu_A(x)|x \in X\}$, here,  
$\mu_A(x)$ is called the membership function (MF) of $x$ in $A$. The membership function maps each element of $X$ to a membership value between 0 and 1.  
![Alt text](fuzzy1.png)



#### G  
**[1] geometric distribution:-**  
This is discrete. It predicts the number of trials needed before we obtain a
success when performing independent trials, each with a known probability p for success. The controlling parameter here is obviously the probability p for success. Real-world examples include estimating the number of weeks that a company can function without experiencing a network failure, the number of hours a machine can function before producing a defective item, or the number of people we need to interview before meeting someone who opposes a certain political bill that we want to pass. Again, for these real-world examples, we might be assuming independence if modeling using the geometric distribution, while in reality the trials might not be independent.   
  
**[2]  Gamma Distribution:-**  
Continuous, has to do with the waiting time until n independent events occur,
instead of only one event, as in the exponential distribution. The gamma distribution is a "waiting time" distribution. Suppose events occur independently and randomly with an average time between events of $\beta$. The waiting time until $\alpha$ events have occurred is a gamma $(\alpha, \beta)$ random variable. The parameter α is known as the shape parameter, and the parameter β is called the scale parameter. Increasing α leads to a more "peaked" distribution, while increasing β increases the "spread" of the distribution.  
**[3] Generalization Error:-**  . Most models
will exhibit a low training loss, but not all of them show a low test
loss. This observation motivates the following definition:  
**Generalization Error= |training loss - test loss|** ,  A trained model is said to generalize well if the generalization error
is small. In linear regresssoin case, the loss is the average squared residual. Thus good generalization means that the average squared residual on test data points is similar to that on the training data. If a model does not generalize well,
then it is said to overfit the training
data.
#### H  
**[1] Hessian Matrix:**-  
is an n×n square matrix composed of the second-order partial derivatives of a function of n variables.  
![Alt text](Hessian.png)  
* If the gradient of a function is zero at some point, that is f(x)=0, then function f has a critical point at x. In this regard, we can determine whether that critical point is a local minimum, a local maximum or a saddle point using the Hessian matrix.  
* If the Hessian matrix is positive definite (all the eigenvalues of the Hessian matrix are positive), the critical point is a local minimum of the function.    
* If the Hessian matrix is negative definite (all the eigenvalues of the Hessian matrix are negative), the critical point is a local maximum of the function.  
* If the Hessian matrix is indefinite (the Hessian matrix has positive and negative eigenvalues), the critical point is a saddle point.  
*  **Note that if an eigenvalue of the Hessian matrix is 0, we cannot know whether the critical point is a extremum or a saddle point.**  
**Convexity and Concavity:-**  
Let $A \subseteq R^n$ be open set and $f:A \to R$, a function whose second derivatives are continuous, its concavity or convexity is defined by the Hessian matrix.  
* Function f is convex on set A if, and only if, its Hessian matrix is positive semidefinite at all points on the set.  
* Function f is strictly convex on set A if, and only if, its Hessian matrix is positive definite at all points on the set.  
*  Function f is concave on set A if, and only if, its Hessian matrix is negative semi-definite at all points on the set.  
* Function f is strictly concave on set A if, and only if, its Hessian matrix is negative definite at all points on the set.   
##### Taylor Polynomial 
$\color{yellow}\Tau(X)=f(a)+(x-a)^T \Delta f(a)+ (x-a)^T\Delta f(a)+\frac{1}{2}(x-a)^T\Eta_f(a)(x-a)+...$  
$\Eta_f(a)$ is a Hessian matrix evaluated at $a$, is the quadratic form that describes how the function curves in the vicinity of $a$.    
### Hilbert space:-  
It is an inner product space that is compelete! functions are linear combination of features.  
A RKHS belonging to a kernel $k(k,x')$ (evaluated at x, centered on x'), $x,x' \in X$ contains functions of the form  
$\color{yellow}f(x)=\sum_{i=1}^m \alpha_ik(x,x_i)$  

 
**Hinge Loss:-**  
for a single data point $(x_i,y_i)$, where $y_i \in ${+1,-1}, the hinge loss is:  
$\color{yellow}L(w,b;x_i,y_i)=max(0,1-y_i(w\cdot x_i +b)) = max(0,1-y_if(x_i))$, so if  
 **[1]** $y_if(x_i)>>1$, i.e., it is correctly classified and far away from boundary points of SVM classifier boundary.  
**[2]** if $y_if(x_i) \le 0$ then it is missclassified. i.e., $y_i=+1$ and $f(x_i)=-3$ in that case max(0,1-(1)(-3))=4 and shows bigger loss.  
 **[3]**
if $0 < y_if(x_i)< 1$, then it is correctly classified but close to the boundary.  
The SVM aims to find a hyperplane that maximizes the margin while minimizing classification errors. This is formulated as minimizing the sum of the hinge loss over all samples, plus a regularization term to control the model's complexity:  
$\color{yellow}J(w,b)=\frac{1}{2}||w||^2+ C \sum_{i=1}^m max(0,1-y_if(x_i))$   
Here, C is a regularization parameter that balances margin maximization and loas minimization.  

**Hyperparameters:-**  Constants whose values are decided by trial and error based on dataset and model are called hyperparameters. Modern ML models have several hyperparameters. Often optimization packages will suggest a default value and a fine-tuning method.

#### I  
**i.i.d:- (independent and identically distributed)**
We assume that the data are generated independently from some unknown (but
fixed) probability distribution $P(x,y)$  
- Our goal is to find a function $f$ that will correctly classify unseen examples $(x, y)$, so that $f(x) = y$ for examples $(x, y)$ that are also generated from $P(x, y)$.

**Infimum or greatest lower bound(glb)-** if set of all lower bounds of set S of real numbers has a greatest member K, then K is said to be a infimum or greatest lower bound of S and it is denoted by Inf S or glb. If the smallest member of a set, if exists is the infimum of a set. the infimum of set of natural number N is 1. because it is bounded below but not above.

#### K  
**A kernel** in machine learning refers to a function that enables algorithms to operate in high-dimensional feature spaces without explicitly mapping data into those spaces. The kernel function is at the heart of this process—it computes the similarity between pairs of data points as if they were mapped into a higher-dimensional (or even infinite-dimensional) space, but does so efficiently by only requiring the original input data.   
The kernelized learning model employing the kernel
matrix, also known as the **kernel trick**, is very amenable to nonvectorial data analysis.The kernel trick can potentially yield an enormous computational saving for the partitioning of highly sparse networks.  
**kernel function** , computes a degree of “similarity” between the objects you are classifying (e.g., images, text documents, or gene expression profiles).

#### L  
**Log-normal Distribution:-**  
This is continuous. If we take the logarithms of each value provided in this
distribution, we get normally distributed data. Meaning that in the beginning,
your data might not appear normally distributed, but if you try transforming
it using the log function, you will see normally distributed data. This is a good distribution to use when encountering skewed data with low mean value, large variance, and assuming only positive values. Just like the normal distribution appears when you average many independent samples of a random variable (using the central limit theorem), the log-normal distribution appears when you take the product of many positive sample values. Mathematically, this is due to an awesome property of log functions: the log of a product is a sum of the logs. This distribution is controlled by three parameters: shape, scale, and location. Real-world examples include the volume of gas in a petroleum reserve, and the ratio of the price of a security at the end of one day to its price at the end of the
day before.  
[2] **Likelihood:-**  
$L(\theta|x)=p_\theta(x) = P_\theta(X=x)$ is a function of $\theta$, is the likelihood function, given the outcome x of the random variable X.  
imagine flipping a fair coin twice,and observing two heads in two tosses ("HH").Assuming that each sucessive coin flip is i.i.d, then probability of observing HH is
$ \color{red}P(HH|p_H=0.5)=0.5^2=0.25$   
Equivalently, the likelihood of observing "HH" assuming $p_H=0.5$  
$ \color{red}L(p_H=0.5|HH)=0.25$, For example we've observed the value of a random variable $X=x$, and we want to know what is/are the value of parameter (i.e., value of pdf/pmf) at that point.  
![Alt text](likelihood1.png)  
if $x=34$ is observed with a mean of 32 and std dev 2.5, the $f(X=34)=0.12$
if we shift the function right a bit with mean of 34 then function value is 0.21. so, in case of MLE (Maximum Likelihood Estimation), we observed values and then tries to find out which parameters (mean,std dev, etc) makes this value maximum.    
**Limit point:-**  limit point $\textit{(or accumulation point)}$ of a set A is defined as a point x such that every open set containing x also contains at least one point of A different from x. In other words, x can be "approximated" by points from the set A. This concept is fundamental in topology and helps in understanding the behavior of sets in a given space.  
More formally, $x$ is a limit point of **A** if all $\epsilon > 0$, there is a point $y \in S \diagdown  \{x\}$ with $d(x,y) < \epsilon$ .  
**Linear Operator:-**  A mapping $L:X \to Y$ between two linear spaces over the same scalar field is called linear operator if it preserves addition and scalar multiplication. This is equivalent to   
$ L(a_1x_1+...+a_nx_n) = a_1L(x_1)+...+a_nL(x_n)$
for all $x_1,...,x_n \in X$, all scalars $a_1,...,a_n$ and all finite $n$. An operator that does not satisfy the above is called nonlinear. The null space and range of L are defined as  
Null(L)= ${x \in X: L(x)=0}$, Range(L)= ${L(x):x\in X}$
#### M  
**[1] membership Function:-** defines the degree to which an element belongs to a fuzzy set, ranging from 0 (not a member) to 1 (full member). It maps elements of a universe of discourse (the set of all possible values) to a membership value between 0 and 1, indicating the extent of their membership in a fuzzy set.   
**[2] Moment:-**   the n-th moment of a distribution is the expected value of the n-th power of the deviations from a fixed value.   
 $k^{th}$ moment of a random variable $X$ with a density function $f(x)$ ,
 
$\color{brown}\mu_k= E[X^k]=\int_{-\infin}^{\infin}  x^k f(x) dx$,  first moment is a random variable’s mean, the second (central moment) is its
variance, and so forth.  
[3] **Markov's Inequality:-**  Markov's inequality provides an upper bound on the probability that a **non-negative random variable**, X, exceeds a positive constant 'a', given by the $P(X \ge a) \le  \frac{E[X]} {a}$  . This means that if the expected value (mean) of a non-negative variable is known, we can estimate the maximum possible probability that the variable will be significantly larger than its average, even without knowing the specific probability distribution.  
- Example:  A biased coid with probability of Head is 1/5 is tossed 10 times, Estimate the probability of getting at least 8 heads.
$E[X]=np=10*(1/5)=2$, by markove inqeuality 
$P(X\ge 8)\le \frac{E[X]}{8}=2/8=1/4$ , but when we calculate actual probability   
$P(X \ge 8) = \sum_{k=8}^{10} \binom{10}{8} (1/5)^8(4/5)^2=0.00007372$,   
- so the question is why markov inequality gives exaggerated values? The upper bound provided can sometimes be quite loose or high because the inequality makes very few assumptions about the distribution, but it will always be true.
#### N  
[1] Normal Distribution:  
    *  PDF for one Random Variable:  $g(x;\mu,\sigma) \frac{1}{ (2\pi\sigma^2)^{0.5}} e^\frac {-(x-\mu)^2}{2\sigma^2}$  
    * PDF for bivariate Normal Distribution: $g(x,y,\mu_1,\sigma_1,\mu_2,\sigma_2,\rho$)=  
    $ \frac {1}{(2\pi^2 det\begin{bmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2  & \sigma_2^2 \end{bmatrix} )^{0.5}}$ $e^{-   \frac {1}{2} \begin{bmatrix}x-\mu_1 & y-\mu_2\end{bmatrix} \begin{bmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2  & \sigma_2^2 \end{bmatrix}^{-1}  \begin{bmatrix} x-\mu_1  \\ y-\mu_2\end{bmatrix} }$  
    where $\rho\sigma_1\sigma_2= Cov(x,y)$   
**[2] Null Space (kernel):-** The set of vectors that yield zero when multiplied by a matrix. The null space or kernel of m x n matrix A is  
$\color{orange}N(A)=ker(A)=\set{x \in F^N: Ax=0}\subseteq F^N $ , for ex.  
$A=\begin{bmatrix}
   1 & 3 \\
   2 & 6
\end{bmatrix}$ , $\implies N(A)=span(\begin{bmatrix}
   -3 \\
   1 
\end{bmatrix})$, and $N^ \bot= span(\begin{bmatrix}
   1 \\
   3 
\end{bmatrix})$
#### O     
[1] **One-hot Encoding:-**  One-hot encoding is a technique used in machine learning and data preprocessing to convert categorical variables into a binary format that algorithms can process effectively. It is particularly useful when dealing with non-numeric data, such as labels or categories.  
[2] **outer product of a vector ($uu^T$)**: if $u$ is a n X 1 column vector then $uu^T$ is a n X n matrix, $(uu^T)_{ij}=u_iu_j$.  
  - **properties:**  
  -> Rank: for any nonzero vector $u$ the matrix $(uu^T)$ has rank 1. all columns and rows are scaler multiple of $u$.  
  -> Symmetry: Matrix is symmetric. i.e., $(uu^T)^T$ =$(uu^T)$  
  -> idempotency: if $u$ is a unit vector , then $(uu^T)^2$= $(uu^T)$   
  -> positive semidefinite: $(uu^T)$ is always positive semidefinite; for any vector $x$,  $x^Tuu^Tx=(u^Tx)^2 \ge0$  
  -> Linearity:  The outer product is linear in both arguments, for example $(cu)u^T= c(uu^T)$  
  -> Geometric meaning: $(uu^T)$ acts as a projector onto the direction of $u$. when applied to any vector $x$, $uu^Tx= u(u^Tx)$, giving a vector in the direction of $u$ whose length is the projection of $x$ onto $u$.
#### P  
**[1] poisson distribution:**  
This is discrete. It predicts the number of rare events that will occur in a given period of time. These events are independent or weakly dependent, meaning that the occurrence of the event once does not affect the probability of its next occurrence in the same time period. They also occur at a known and constant average rate $\lambda$. Thus, we know the average rate, and we want to predict how many of these events will happen during a certain time period. The Poisson distributions controlling parameter is the predefined rare event rate $\lambda $. Real-world examples, include predicting the number of babies born in a given hour,the
number of defective items produced by a certain machine on a certain day, the
number of people entering a store at a certain hour, the number of car crashes an insurance company needs to cover within a certain time period, and the number of earthquakes happening within a particular time period.  
[2] **P-value:-**   
- is the probability of obtaining a sample statistic as extreme as the one observed, assuming the null hypothesis is true. For example, if a factory claims that their tires have a mean weight of 200 pounds, and an auditor finds a p-value of 0.04, it means that there is a 4% chance of observing such a difference due to random sampling error if the factory's claim is true.  **p-values are calculated under the assumption that the null hypothesis is true.**    
- Statistical significance is determined by comparing the p-value to a predefined significance level (alpha). Common significance levels are 0.01, 0.05, and 0.10. If the p-value is less than the significance level, the null hypothesis is rejected, indicating that the results are statistically significant.  
[3] **Precision:-** it is a ratio of correct positive predictions to overall number of positives.  so, precision =$\color{darkblue}\frac{TP}{TP+FP}$

#### R    
**Rank of a Matrix:-**  
The rank of a matrix is the maximum number of linearly independent rows or columns in the matrix. It essentially represents the dimension of the vector space spanned by the matrix's rows or columns. **In simpler terms, it indicates the number of independent equations represented by the matrix**.
  
**Recall:-**, it is a ratio of correct positive predictions to the total number of positive examples. recall = $\color{darkblue}\frac{TP}{TP+FN}$

**ROC (Receiver Operating Characteristics) Curve:-**    
It uses a compbination of:   
**TPR(True Positive Rate):** the proportion of positive example predicted correctly. (i.e., recall).  
**FPR(False Positive Rate):** The proportion of negative examples predicted incorrectly.  
TPR=$\color{darkblue}\frac{TP}{TP+FN}$   , FPR=$\color{darkblue}\frac{FP}{FP+TN}$


**Random Vector:**- it is a collection of Random variables.  

$ \color{yellow} X=\begin{bmatrix}
   x1  \\
   x2 \\
   .. \\
   x_n
\end{bmatrix}$, now, if $q$ is a unit vector of n-dimention and onto which **X** to be projected This projection is defined by the inner product of the vectors X and q, as shown by $ \color{yellow}A=q^TX = X^Tq$,   
now,a projection matrix $P=qq^T$, projects any vector onto the line spanned by $q$. $||q||=(qq^T)^{1/2}=1$.  
The projection A is a random variable with a mean and variance related to the statistics of the random vector **X *. When data (assumed centered) is projected onto the line defined by q, the variance of the projected data is given by the quadratic form:  $ \color{yellow}V_q=q^TCq$. where $C$ is covarience matrix.$V_q$ measures how much the data spreads along the direction q after projection.  
As q varies over all possible unit vectors, $V_q$ changes because the projection matrix $P=qq^T$ changes accordingly, and so does the direction along which the variance is measured. if $q$ is aligned with the eigenvectors of $C$ (principal component), the $V_q$ is maximized (or minimized) and equals the corresponding eigenvalue of $C$.   
The projection A is a random variable with a mean and variance related to the statistics of the random vector X.  
$\color{yellow}\sigma^2= E[A^2]=E[(q^TX)(X^Tq)]=q^TE[XX^T]q=q^TRq$ ...  (eq..r1),  
 where $R$ is Correlation matrix, defined as the expectation of the outer product of the vector X with itself. it is symmetric i.e., $R^T=R$.  
 The variance $\sigma^2$ of the projection $A$, is the function of unit vector $q$, we may say $\color{yellow}\psi(q)=\sigma^2=q^TRq$ 


#### S    
**supremum or least upper bound:-**  if a set of all upper bounds of a set S of real numbers has a smallest member k, then k said to be least upper bound or supremum of S and it is denoted by lub or Sup S. if supremum of infimum are unique if they exist.   The necessary and sufficient condition for a real number 's' to be supremum of a bounded above set **S** is that 's' must satisfy the following conditions.  
[1] $x\le s, \forall x \in S$  
[2] for each positive real number $\epsilon$, $\exist$ a real numer $x\in S$, such that $x > s-\epsilon$.  
greatest member of a set if exists, is the supremum or lub of the set.

 $\color{blue}\Sigma^{-1}$:-  sigma inverse in gaussian equation.  
 sigma inverse $\color{yellow}\Sigma^{-1}$ in the multivariate Gaussian is the inverse of the covariance matrix, and it determines how deviations from the mean are weighted based on the variance and covariance structure of the data  
**[1] Standard Error (SE):-**   
  - A statistical metric that shows how much variation there might be between different samples of a population and the true poppulation.
  - the standard deviation of $\bar x$ determines the amount of sampling error to be expected when a population mean is estimated by a sample mean, it is often referred to as the standard error of the sample mean. In general, the standard deviation of a statistic used to estimate a parameter is called the standard error (SE) of the statistics.   
- Roughly, the standard error of the estimate indicates how much, on
average, the predicted values of the response variable differ
from the observed values of the response variable.  
- SE of estimates:-  $S_e= ({SSE \over n-2})^{1 \over 2}$,  where SSE is error sum of squares = $\sum_{i=1}^n (y_i- \hat y)^2$   

**[2] $S_{xx}$**   
$S_{xx}= \sum_{i=1}^n(x_i-\bar x)^2 = \sum x_i^2-2\bar x\sum x_i+ \sum \bar x^2 $  
since $\sum x_i=n \bar x$ and $\sum \bar x^2=n \bar x ^2$  
$\sum x_i^2-n\bar x^2$, also equals to $\sum x_i^2- \frac{\sum x_i^2}{n}$  


**[3] Spectral Theorem:-** it identifies conditions under which a linear operator or matrix can be diagonalized.The theorem tells us when a matrix (or linear operator) can be transformed into a diagonal matrix using an invertible matrix (or unitary operator).  

**[4] Sufficient Statistics:-** A sufficient statistic is a function of sample data (e.g., a summary of the data, such as a sum or a mean) that contains
all the information needed to estimate an unknown parameter in a statistical model.
#### T  
**Taylor Series:-**  
if a function $f(x)$ is a sum of power series about $x=a$ , $f(x)= \sum_{n=0}^\infin a_n(x-a)^n $  
$= a_0+a_1(x-a)+a_2(x-a)^2+...+a_n(x-a)^n+...$,  
with a positive radius of convergence. By repeated term-by-term differentiation within the 
interval of convergence $I$, we obtain  
$f'(x)= a_1+2a_2(x-a)+ 3a_3(x-a)^2+...+ na_n(x-a)^{n-1}+...$  
$f''(x)= 1\cdot2a_2+2\cdot3a_3(x-a)+3\cdot4 a_4(x-a)^2+...$  
with the $n^{th}$ derivative being $f^{a_n}(x)+ n!a_n+$ a sum of terms with $(x - a)$ as a factor.   
**$f'(a)=a_1, f''(a)=1\cdot2a_2, f'''(a)=1\cdot2\cdot3a_3$, in general $f^{(n)}(a)=n!a_n$   
then $a_n$= $\frac{f^{(n)}}{n!}$**   ,now our definition for Taylor Series is ..  
--> Let ƒ be a function with derivatives of all orders throughout 
some interval containing a as an interior point. Then the Taylor series generated  by ƒ at x = a is  
$\sum_{k=0}^\infin \frac{f^{(k)}(a)}{k!}(x-a)^k= f(a)+\frac{f'(a)}{1!}(x-a)+ \frac{f''(a)}{2!}(x-a)^2+...+\frac{f^{(n)}(a)}{n!}(x-a)^n+...$  
**Taylor Polynomials:-**
The linearization of a differentiable function $f$ at a point a is the polynomial of degree one given by $P_1(x)=f(a)+f'(a)(x-a)$

#### V  
**Variance and Standard Deviation:**- The sample variance is denoted by $s^2$ is the sum of square deviation from mean divided by n-1. it is based of observed data.  
**Definition:  Variance** measures how much the predicted values fluctuate around their average prediction. It reflects the sensitivity of the model predictions to different training data sets.  
$s^2= \frac {\sum (x-\bar x)^2}{n-1}= \frac {S_{xx}}{n-1}$  
The sample standard deviation is the positive square root of the sample
variance and is denoted by s. Division by $n-1$  ensures that it is an unbiased estimator of the population variance. An alternative rationale for using $(n-1)$ is based on the property $\sum(x-\bar x)=0$,
Suppose that n=5 and that four of the deviations are  
$x_1-\bar x=-4$, $x_2-\bar x=6$,$x_3-\bar x=1$,$x_4-\bar x=-8$, then our $x_5-\bar x$ must be 5. Although there are five deviations, only
four of them contain independent information about variability. More generally, once any $n-1$ of the deviations are available the value of remaining deviations are determined. The n deviations actually contain only (n-1) independent pieces of information about variability. Statisticians express this by saying that $s^2$ and $s$ are based on (n - 1) degrees of freedom (df).
  * **Random Variable Variance ($\sigma^2$)**: It describes properties of a probability distribution, such as how concentrated or dispersed values are around the mean. Used in probability theory and theoretical statistics to analyze distributions. it is based on theoretical probability. distribution. 
  $Var(X)= E[(X=\mu)]= \Sigma(x-\mu)^2p_x(x)$
  poopulation variance also is denoted by $\sigma^2$ and std.dev by $\sigma$.   
    ##### another definition of variance:-   
    Suppose x is a scalar random variable with probability density function f(x); to make the discussion below more relevant x can be thought of as a coefficient estimate $\hat \beta$ and thus $f(\hat \beta)$ would be its sampling distribution. The variance of x is defined as  
$ \color{yellow}V(x)=E(x-Ex)^2= \intop(x-Ex)^2f(x)dx$  
- - In words, if you were randomly to draw an x value and square the difference between this value and the mean of x to get a number Q, what is the average value of Q you would get if you were to repeat this experiment an infinite number of times? An alternative description in words is as follows: a weighted average, over all possible values of x, of the squared difference between x and its mean, where the weights are the "probabilities" ofthe x values.  
- The covariance between two variables, x and y is defined as  
$\color{yellow}C(x,y)=E(x-Ex)(y-Ey)= \iint(x-Ex)(y-Ey)f(x,y)dxdy$, where $f(x,y) $ is a joint density function for x and y.  
**Estimation:-**  
estimate of V(x), which can be calculated if some data are available.  
(a) for x a scalar, if we have N observations n x, say $x_i$ through $x_N$, then V(x) is usually estimated by  
$\color{yellow}s^2=\sum(x_i-\bar x)^2/(N-1)$ , same way covariance $ \color{yellow}C(x,y)$ is  $\color{yellow}s_{xy}=\sum (x_i-\bar x)(y_i-\bar y)/(N-1)$  
- Univariate nonlinear function g(x) of a vector x: asymptotically,  
$V(g(x))= (\partial g/ \partial x)' V(x) (\partial g/ \partial x) $




  **Variational Inference:-**  
  is an important class of approximate inference algorithms; its basic idea is to choose an approximate distribution $q(x)$ from a family of tractable or easy-to compute distributions with trainable parameters and then make this approximation as close as possible to the true posterior distribution $p(x)$.  

  **Vector Projection:-**   
  ![Alt text](vector_project.png)   The length of the residual vector ($\bar r$)
provides a precise measure of “closeness,” where a shorter residual vector ($\bar r = \bar v - x \bar u$) indicates that linear combination $x \bar u$ is closer to target vector $\bar v$ and longer $\bar r$ indicates that ($x \bar u$) is farther from ($\bar r$).  
  -  so, For a vector ($\bar v$) and a nonzero vector ($\bar u$), the projection of ($\bar v$) onto ($\bar u$)
is the vector (x$\bar u$) where  $\color{orange}x= \frac {\bar v \cdot \bar u} {\bar u \cdot \bar u}$,  When u = 0 is the zero vector, the projection of v onto u is 0.  

**Vector Subspaces:-**  Let V be a vector space over the field F and $ W \subseteq V$, then, W is called a subspace of V if W itself is a vector space over F with respect to the operations of vector addition and scaler multiplication. 

#### W  
**[1] Weibull Distribution:-**  
This is continuous. It is widely used in engineering in the field of predicting
product lifetimes (10-year warranty statements are appropriate here as well).
Here, a product consists of many parts, and if any of its parts fail, then the
product stops working. For example, a car will not work if the battery fails,
or if a fuse in the gearbox burns out. A Weibull distribution provides a good
approximation for the lifetime of a car before it stops working, after accounting
for its many parts and their weakest link (assuming we are not maintaining the
car and resetting the clock). It is controlled by three parameters: shape, scale,
and location. The exponential distribution is a special case of this distribution,
because the exponential distribution has a constant rate of event occurrence, but the Weibull distribution can model rates of occurrence that increase or decrease
with time.

#### Z  
* **Z-Score**
  The z-score corresponding to a perticular value is   
  $z-score = \frac {value-mean}{standard deviation}$  
  The z score is particularly useful when the distribution of observations is approximately normal.The z score tells us how many standard deviations the value is from the mean. It is positive or negative according to whether the value lies above or below the mean. The process of subtracting the mean and then dividing by the standard deviation is
sometimes referred to as standardization, and a z score is one example of what is called a standardized score.  
- It is used to get confidence interval. For ex. for 95% of confidence interval if we want to assess confidence interval for a coefficients...  
$\color{blue}\hat \beta_1 \pm z_{1-\alpha/2} \hat {SE} (\hat \beta_1)$  
so, we should find z-score for $\alpha=(1-0.95)=0.5, \alpha/2=0.025$  
Area under the curve to left $A_L= \frac{1+confidence level}{2}=\frac{1.95}{2}=0.975$.  
- now find 0.97500 in z table that is 1.96.   
- in Excel $=NORM.S.INV(1-(1-0.95)/2)=1.959964=1.96$

The Pythagorean theorem is a fundamental relation in Euclidean geometry between the three sides of a right triangle. It states

> the area of the square whose [side] is the [hypotenuse](https://en.wikipedia.org/wiki/Hypotenuse) (the side opposite the right angle) is equal to the sum of the areas of the squares on the other two sides.

::::div{.grid .grid-cols-5}

:::div{.col-span-2 .text-center}
## Equation

The theorem can be written as an equation relating the lengths of the sides $a$, $b$ and the hypotenuse $c$

$$
a^{2}+b^{2}=c^{2}
$$

:::

:::div{.col-span-3}
![](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Pythagorean.svg/200px-Pythagorean.svg.png){.mx-auto .mt-20}
:::
