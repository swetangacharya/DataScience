---
title: Pattern Classification (Template)
description: Template post on how to boil a pot of water. Presumably this is for cooking.
slug: how-to-boil-water
is_draft: true
icon: cooking-pot
tags:
  - Parzen window
  - cooking tutorials
  - basic life skills
---

:::info
**New here?**  
Every post begins with a random template to help you start writing.

**Still confused?** 🤔  
- [:icon[video] Watch the **Getting Started** video :icon[external-link]](https://www.youtube.com/watch?v=0h4gRvgoRn4&list=PL9Zhnnyw1lVND99JOWpTyYly9heBP4-Xh){target=_blank}
- [:icon[file-text] Read the **Getting Started** guide](https://www.scipress.io/post/l7R0XuDTe6R1dC2dS5cc/Getting-Started)
:::

---
#### Non-Parametric Techniques: (duda)  
There are several types of nonparametric methods of interest in pattern recognition. One consists of procedures for estimating the density functions $P(x|w_j)$ from
sample patterns. If these estimates are satisfactory, they can be substituted for the
true densities when designing the classifier. Another consists of procedures for directly estimating the a posteriori probabilities $P(w_j|x)$. This is closely related to nonparametric design procedures such as the **nearest-neighbor rule**, which bypass probability estimation and go directly to decision functions. Finally, there are nonparametric procedures for transforming the feature space in the hope that it may be possible to employ parametric methods in the transformed space. These discriminant analysis methods include the Fisher linear discriminant.  
  ##### Density estimation  
 - **Parzen Window:-**  The Parzen-window approach to estimating densities can be introduced by temporarily assuming that the region $\R_n$ is a d-dimensional hypercube. If $h_n$ is the length of an edge of that hypercube, then its volume is given by  $\color{purple}V_n=h_n^d$

  - The basic ideas behind many of the methods of estimating an unknown probability density function are very simple, although rigorous demonstrations that the estimates converge require considerable care. The most fundamental techniques rely on the fact that the probability $P$ that a vector **x** will fall in a region $\R$ is given by   
   $\color{purple}P = \int_R p(x')dx'$     
   Analytic expression for the number of samples falling in the hypercube ($k_n$), with window function $\color{purple}\phi(u) = \begin{cases}
   1 &\text{if } |u_j| \le 1/2 ,\ j=1,...,d \\
   0 &\text{otherwise}
\end{cases}$  
- Thus, $\phi(u)$  defines a unit hypercube centered at the origin. It follows that $\phi((x-x_i)/h_n)$ is equal to unity if **$x_i$** within hypercube of volume $V_n$ centered at **x**, and is zero otherwise.  
The number of samples in this hypercube is given by  
$\color{purple}k_n= \sum_{i=1}^n \phi (\frac{x-x_i}{h_n})$   
- let $V_n$ be the volume of $\R_n$, $k_n$ be the number of samples falling within $\R_n$, and $p_n(x)$ be the nth estimate of $p(x)$ is ,  
 $\color{purple}p_n(x)= \frac{k_n/n}{V_n}$   
 - these n samples $x_1,...,x_n$ are drawn i.i.d, then k of these n fall in region $\R$ is   
$\color{purple}P_k= \dbinom{n}{k}P^k (1-P)^{n-k}$  
$\color{purple}P_n(x)= \frac{1}{n}\sum_{i=1}^n  \frac{1}{V_n}\phi(\frac{x-x_i}{h_n})$   , is the estimated density at **x**, $V=h^D$, volume of a hypercube of side h in D dimentions. this can be interpreted as, not a single cube centered on **x**, but it is a the sum over N cubes centered on N data points $x_n$.   the window function is being used for interpolation — each sample contributing to the estimate in accordance with its distance from x.  
- In below figure, We see that h acts as a smoothing parameter and that if it is set too small (top panel), the result is a very noisy density model, whereas if it is set too large (bottom panel), then the bimodal nature of the underlying distribution from which the data is generated (shown by the green curve) is washed out. The best density model is obtained for some intermediate value of h (middle pane).  
![Alt text](parzen_window1.png) 
   We see that, as expected,
the parameter h plays the role of a smoothing parameter, and there is a trade-off between sensitivity to noise at small h and over-smoothing at large h. Again, the optimization of h is a problem in model complexity, analogous to the choice of bin width in histogram density estimation, or the degree of the polynomial used in curve fitting. we can choose any kernel function $\phi(u)$, subject to   

$\color{purple}\phi(u) \ge 0$,  
$\color{purple}\int \phi(u)du = 1$  
Let us look at below example:  Suppose we have two training data points located at 0.5 and 0.7, and we use 0.3 as its rectangle window width. Using the Parzen window technique, what would be the probability density if we assume the query point is 0.5?  
our case n=2, h=0.3 and x=0.5, h/2=0.15 then x+/-h/2= [0.15,0.65], 0.5 lies inside this window and 0.7 is outside.
so $\hat p(0.5)= \frac{1}{n}\sum_{i=1}^n \frac{1}{h} \phi(\frac{x-xi}{h})$  
$\phi(u)=1$ if $|u| \le\frac{1}{2}$ , our x=0.5 (query point) as we want to find $\hat p(x=0.5)$ . $x_1=0.5, x_2=0.7$,  
 so, first value is (0.5-0.5)/0.3=0 , and $\phi(0)=1$  
and second value is (0.5-0.7)/0.3=-0.67 and $\phi(-0.67)=0$ (because $|-0.67| \\> \frac{1}{2} $)  

Then, $\color{magenta}\hat p(0.5)=\frac{1}{2}*\frac{1}{0.3}*(1+0)= 1.66$

- As we have seen, one of the problems encountered in the Parzen-window/PNN approach concerns the choice of the sequence of cell volumes sizes $V_1,V_2...$ or overall window size. for ex. if we take $V_n=V_1/n^{0.5}$, the results for any finite n will be very sensitive to the choie for the initial volume $V_1$. Furthermore, it may well be the case that a cell volume appropriate for one region of the feature space might be entirely unsuitable in a different region. Now, we turn to an important alternative method that is both useful and has solvable analytic properties.  
#### $K_n$-Nearest Neighbour Estimation:-   
A potential remedy for the problem of the unknown “best” window function is to let the cell volume be a function of the training data, rather than some arbitrary function of the overall number of samples. For example, to estimate
$p(x)$ from $n$ training samples or prototypes we can center a cell about **x** and let it grow until it captures $k_n$ samples, where $k_n$ is some specified function of $n$. These samples are the $k_n$ nearest-neighbors of **x**. It the density is high near **x**, the cell will be relatively
small, which leads to good resolution. If the density is low, it is true that the cell will grow large, but it will stop soon after it enters regions of higher density. In either case, if we take  
  $\color{magenta}p_n(x)= \frac{k_n/n}{V_n}$  
we want $k_n$ to go to infinity as n goes to infinity, since this assures us that $k_n/n$ will be a good estimate of the probability that a point will fall in the cell of volume $V_n$. However, we also want $k_n$ to grow sufficiently slowly that the size of the cell needed to capture kn training samples will shrink to zero. Thus, it is clear from above equation that ratio $\frac{k_n}{n}$ should go to zero.    
#### Ensemble classifiers methods:-
- The methods generally vary the input given to each classifier. These include subsampling the training set, manipulating the features of the training set, manipulating the output target, injecting randomness and some methods which are specific to
particular algorithms
##### Stacking:-   
![Alt text](Stacking.png)

##### Bagging:-  
![Alt text](bagging.png)  
The main intuition of using bagging is to reduce variability in the partitioning results through averaging.  
Bagging(bootstrap agreegating) is a technique that can reduce the variance and improve the generalization error performance. The basic idea is to create B variants, $X_1,X_2,...,X_B$, of the training set X, using bootstrap techniques, by uniformly sampling from X with replacement. For each of the training set variants $X_i$, a tree $T_i$ is constructed. The final decision for classification of a given points is in favor of the class predicted by majority classifiers $T_i, i=1,2,...B$.  

**Random forests** use the idea of bagging in tandem with random feature selection. The difference with bagging lies in the way the decision trees are constructed. The feature to split in each node is selected is the best among the set of F randomly chosen features, where F is a user-defined parameter. This extra introduced randomness is reported ot have a substantial effect in performance improvement.
Random forests, often have very good predictive accuracy and have been used in a number of application including body pose recognition. 

##### Boosting:-  
![Alt text](boosting1.png)   
**AdaBoost:- (An algorithmic approach)**  
The general problem of producing a very accurate prediction rule by combining through and moderately inaccurate rules-of-thumb is referred to as boosting. The booster is provided with a set of labelled training examples $(x_1, θ_1), ...,(x_N, θ_N )$, where $θ_i$ is the label associated with instance $x_i$. On each round $t = 1, ..., T$, the booster devices
a distribution $D_t$ over the set of examples and uses a weak hypothesis $h_t$ having low error $\epsilon_t$ with respect to $D_t$. Any classifier can be used at this point. The weak classifier can be a **decision stump**—a decision tree with depth one. In other words, the classification is based on a single decision node. Thus, distribution $D_t$ specifies the relative importance of each example for the current round. After T rounds, the booster must combine the weak hypothesis into a single prediction rule.

  **ADABOOST** maintains the probability distribution $p_t(x)$ over the training set. In each iteration t, it draws a training set of size m
by sampling with replacement according to the probability distribution $p_t(x)$. The classifier is used on this training set. The error rate $\epsilon_t$ of this classifier is computed
and used to adjust the probability distribution on the training set. The probability
distribution is obtained by normalising a set of weights, $w_t(i), i = 1, .., n$ over the
training set. The effect of the change in weights is to place more importance on
training examples that were misclassified by the classifier and less weight on the
examples that were correctly classified. The final classification is constructed by a
weighted vote of the individual classifiers. Each classifier is weighted according to its
accuracy for the distribution $p_t$ that it was trained on.



#### Hidden Markov Model (HMM):-   
- A Markov model for a sequence of random variables $x_1, ..., x_T$ of order 1 is a joint
probability model that can be factorized as follows...  
$p(x_1,...,x_T)= p(x_1)p(x_2|x_1)...p(x_T|x_{T-1}) = p(x_1) \prod_{t=2}^T p(x_t|x_{t-1})$  
- Notice that each factor is a conditional distribution conditioned by 1 random variable
(i.e., each factor only depends on the previous state and, therefore, has a memory of 1).
We can also say that $X_{t-1}$ serves as a sufficient statistic for $X_t$ . A sufficient statistic is a function of sample data (e.g., a summary of the data, such as a sum or a mean) that contains
all the information needed to estimate an unknown parameter in a statistical model.   
- Let's look at the transition probability $p(x_t |x_{t–1}))$ in more detail. For a discrete state
sequence $x_t \in \{1, ..., K\}$, we can represent the transition probability as a K x K stochastic matrix (in which the rows sum to 1).  
$\color{orange}A_{ij}=p(X_t=j|X_{t-1}=i)$ where $\color{orange}\sum_J A_{ij}=1 , \forall i$  
![Alt text](Markov1.png)   
Here, $\alpha$ is the transition probability out of state $1$ and $1-\alpha$
is the probability of staying in state 1. Notice how the transition probabilities out of each state add up to 1. We can write down the corresponding transition probability matrix as follows  
A=$\begin{pmatrix}
   1-\alpha & \alpha \\
   \beta & 1-\beta
\end{pmatrix}$  
- If $\alpha$ and $\beta$ do not vary over time; in other words, if the transition matrix A is independent of time, we call the Markov chain stationary or time invariant. We are often interested in the long-term behavior of the Markov chain—namely, a distribution over states based on the frequency of visits to each state as time passes.
Such distribution is known as stationary distribution. Let's compute it! Let $\pi_0$ be the initial distribution over the states. Then, after the first transition, we have the following.  
 $\color{orange}\pi_1(j)= \sum_i \pi_0(i) A_{ij}$  
 We can keep going and write down the second transition as follows.  
  $\color{magenta}\pi_2= \pi_1 A = \pi_0 A^2$
- We see that raising the transition matrix A to a power of n is equivalent to modeling n
hops (or transitions) of the Markov chain. After some time, we reach a state when left
multiplying the row state vector $\pi$ by the matrix A gives us the same vector $\pi$.  
$\color{magenta}\pi=\pi A$  
In the preceding case, we found the stationary distribution; it is equal to $\pi$, which is an
eigenvector of A that corresponds to the eigenvalue of 1. We will also state here 
 that a stationary distribution exists if and only if the chain is
recurrent (i.e., it can return to any state with probability 1) and aperiodic (i.e., it doesn’t
oscillate).

  

- **Evaluating classifier.**
![Alt text](evaluation_classifier1.png)  

- ROC Curve (Receiver Operating Characteristic Curve)  
![Alt text](ROC1.png)

##### Latent Variable Model (as a Generative Model)(BS IIT Madras):

Data set $D={x_1,x_2,...,x_n}$ is drawn from unknown distribution of $x$, $P_x$, and $P_{\theta}$ is a parametric model,
Now, in case of Latent Variable model it is defined as  $P_{\theta}(x)=\sum _z P_{\theta}(x,z)$ or $P_{\theta}(x)=\int_z P_{\theta}(x,z)dz$, where is z is latent or hidden or unobserved R.V. it gives some extra information about the data that is not observed while collecting the data, where x is observed variable.
Usually, latent variable $z$ is jointly estimated along with model parameters $\theta$, for example in case of gaussian model $\theta$ can be $\mu$ and $\sigma$. So, in latent variable model we estimate z and {\theta} togather.
Now, for each $x_i \in D$, we assume that there is corresponding $z_i$ exist.  
- **[case 1] when z is discrete**.i.e., $z \in {1,2,...,M}$ and $x \in \R^d$, hence $z_i|x_i$, represents latent variable corresponding to $x_i$.
we can think that each $x_i$ is put into the bucket of M categories of the $z_i$. This is nothing but clustering algorithm, ex. **GMM,k-means**.
so, we're grouping the $x_i$ in on of the M latent variable categories.  
- **case[2] when z is continuous:-**, $x \in \R^d, z \in \R^k, {k<<d}$, Autoencoder is the example of this. Here, $z_i|x_i$ represent a feature-vector correspond to given $x_i$. because dimention of z is smaller than that of **x**; latent variable z can be used to represent data in lower dimensions.


 ##### K-means Clustering:-
 

* The **K-Nearest Neighbour**, faces some important challenges that limit  its performance. The classifier is sensitive to noise and outliers, and requires that  the training data be stored and processed continuously.  
  1. The classifier treats equally all entries (i.e., attributes) of the feature vec
tor. If, for example, some attributes are more relevant to the classification
 task than the remaining attributes, this aspect is ignored by the k-NN
 implementation because all entries in the feature vector contribute simi
larly to the calculation of Euclidean distances and the determination of
 neighborhoods.  
  2. The k-NN classifier does not perform well in high-dimensional feature
 spaces when M is large for at least two reasons:-  
    *  First, for each new feature h, the classifier needs to perform a search  over the entire training set to determine the neighborhood of h. This step is demanding for large M and N.
    * Second, and more importantly, in high-dimensional spaces, the training
 samples ${\{h_n}\}$ only provide a sparse representation for the behavior of
 the data distribution $f_{r,h},(r,h)$. The available training examples need
 not be enough for effective learning.

 Partition of Dataset D into k subsets is called clustering, $C_1,C_2...C_k$.   
$\color{magenta}\bigcup_{i=1}^k C_i=D $  and $\color{magenta}\bigcap_{i=1}^k C_i= \phi$.  
A “good” cluster is a subset of points which are closer to the **mean** of their own cluster than to the **mean** of other clusters.  
mean=  $\color{green}\vec {y_i}= \frac{1}{m_i}\sum_{\vec x \in C_i} \vec x$   
variance= $\color{green}\sigma^2 = \frac{1}{m_i} \sum_{\vec{x} \in C_i} ||\vec{x}-\vec{y_i}||_2^2$.  
 Given k, the desired number of clusters, the
k-means clustering partitions $D$ into $k$ clusters $C_1, C_2, . . . , C_k$ so as to minimize the cost function.  
$\color{green}\sum_{i=1}^{k}\sum_{\vec{x} \in C_i} ||\vec{x}-\vec{y_i}||_2^2$.    
**Cost of $C_i$ is $\color{green}\sum_{\vec{x} \in C_i} ||\vec{x}-\vec{y_i}||_2^2$.**  

Maintain clusters $C_1,C_2,...,C_k$  
For each cluster $C_i$,   
find the mean $\vec y_i$
Initialize new cluster $C_i^{'} \gets \phi$  
for $\vec x \in D$ do   
	$i_x=argmin_i ||\vec x - \vec y_i||_2$  
	$C_{i_x}^{'} \gets C_{i_x}^{'} \bigcup \{\vec x\}$  
end for  
Update clusters $C_i \gets C_i^{'}$


- At each iteration, we find the mean of each current cluster. Then
for each data point, we assign it to the cluster whose mean is the
closest to the point, without updating the mean of the clusters. In
case there are multiple cluster means that the point is closest to, we
apply the tie-breaker rule that the point gets assigned to the current
cluster if it is among the closest ones; otherwise, it will be randomly
assigned to one of them. Once we have assigned all points to the new
clusters, we update the current set of clusters, thereby updating the
mean of the clusters as well. We repeat this process until there is no
point that is mis-assigned.

-  **Disadvantage:-**  (Python Data science Handbook)

    - One way to think about the k-means model is that it places a circle (or, in higher dimensions, a hyper-sphere) at the center of each cluster, with a radius defined by the most distant point in the cluster. This radius acts as a hard cutoff for cluster assignment within the training set: any point outside this circle is not considered a member of the cluster.
    - for k-means is that these cluster models must be circular: k-means has no built-in way of accounting for oblong or elliptical clusters. So, for example, if we take the same data and transform it, the cluster assignments end up becoming muddled. See the two figures below, one is before transform and another one is after transform.
    ![Alt text](k-means1.png)  ![Alt text](k-means2.png)  


**- Gaussian Mixture Models:- (GMM)**    
- A Gaussian mixture model (GMM) attempts to find a mixture of multi-dimensional Gaussian probability distributions that best model any input dataset. In the simplest case, GMMs can be used for finding clusters in the same manner as k-means. KMeans uses a distance-based approach, and GMM uses a probabilistic approach. There is one primary assumption in GMM: the dataset consists of multiple Gaussians, in other words, a mixture of the gaussian.  
- In the beginning, we didn't have any insights about clusters nor their associated mean and covariance matrices
Well, It happens according to the below steps,

[1] Decide the number of clusters (to decide this, we can use domain knowledge or other methods such as BIC/AIC) for the given dataset. Assume that we have 1000 data points, and we set the number of groups as 2.  
[2] Initiate mean, covariance, and weight parameter per cluster. (we will explore more about this in a later section)  
[3] Use the Expectation Maximization algorithm to do the following,  
:::info[steps]
  - Expectation Step (E step): Calculate the probability of each data point belonging to each data point, then evaluate the likelihood function using the current estimate for the parameters.  
  -  Maximization step (M step): Update the previous mean, covariance, and weight parameters to maximize the expected likelihood found in the E step.  
  - Repeat these steps until the model converges.
:::
:::info[model Assumptions ]

 - 1-hot-encoded discrete latent variables $z_k \in {0,1}$, for K clusters, with prior    	$p(z_k=1)=\pi_k, \pi_k \in [0,1],  \sum_{i=1}^k \pi_i=1  $
 - The clusters are Gaussians, with different parameters
 	$p(x|z_k=1)=\Nu(x|\mu_k,\sum_k)$  
 -  It follows joint distribution, $p(x,z_k=1)=p(x|z_k=1)p(z_k=1)=\pi_k\Nu(x|\mu_k,\sum_k)$  
 - And the marginal... the **full generative model**
 $p(x)=\sum_{\substack{z}} p(x,z)= \pi_k\Nu(x|\mu_k,\sum_k) $
:::  

Example:- Consider a one-dimensional dataset:  x=[1,2,3,4,5]
 Model this data using a GMM with two components  K=2 
 Initial parameters are  
component 1: $\pi_1=0.5,\mu_1=2, \sigma_1=1$  
component 2: $\pi_1=0.5,\mu_1=4, \sigma_1=1$

A normal Gaussian PDF is ...  
$\color{brown}\Nu(x|\mu,\sigma^2)= \frac{1}{(2\pi \sigma^2)^{0.5}} exp \lparen \frac{-(x-\mu)^2}{2\sigma^2}  \rparen$

the responsibility $\gamma_{ik}$ for data point $x_i$ belonging to component $k$ is calculated as 
$\color{brown} P(j|x_i)=\gamma_{ik}= \frac{\pi_k \Nu(x_i|\mu_k,\sigma_k^2)}{\sum_{j=1}^K \pi_j \Nu(x_i|\mu_j,\sigma_j^2)}$    ... (1)  
- A **responsibility** is the posterior probability that a given data point was generated by a specific Gaussian component. It represents a soft assignment, indicating the extent to which a data point belongs to each cluster or distribution within the mixture, with the sum of responsibilities for a single point always equaling one.  => Since the class labels are not known, for each data value we can only determine the probability that it was generated by class j (sometimes called responsibility,Given xi , this probability can be obtained for each class using Bayes' rule. The class probability $p(j|x_i)$ is small when $x_i$ is not within “a few” $\sigma_j$ (Std. dev) from $µ_j$ (assuming that $x_i$ is close to some other mixture component). Of course, $\sum_{j=1}^k p(j|x_i) = 1$

class1 weighted porbabilities= [0.121, 0.1994, 0.121, 0.027, 0.0022]  
class2 weighted probabilities= [0.0022, 0.027, 0.121, 0.1994, 0.121]  

- new new resposibilities for $x_1=1$ and class 1 and class 2 i.e., $\gamma_{11}$ and $\gamma_{12}$  
$\gamma_{11}= \frac{0.121} {0.121+0.0022}=0.982$ , $\gamma_{12}= \frac{0.0022} {0.121+0.0022}=0.0178$, $\gamma_{12}= 1-\gamma_{11}$ and viceversa.  
class 1 responsibilities= [0.9821, 0.8807, 0.5, 0.1193, 0.0179]   
class2 responsibilities= [0.0179, 0.1193, 0.5, 0.8807, 0.9821]  
Now calculate totatal responsibility of class1
$N_1= \sum_{i=1}^{5} \gamma_{i1}=0.989+0.881+0.5+0.119+0.011=2.5$   
sameway, total responsibility of class 2  
$N_2= \sum_{i=1}^{5} \gamma_{i1}= 0.0179+ 0.1193+0.5+ 0.8807+ 0.9821=2.5$       
now, new mean is ...  
$\mu_1^{new}= \frac{1}{N_1} \sum_{i=1}{5} \gamma_{i1} = \frac{1}{2.5} (0.9821 *1 + 0.8807*2+0.5*3+ 0.1193*4+0.0179*5)=1.924 $   
finally, updated variance is calculated as   
$\sigma_1^{new}\approx \frac{1}{N_1} \sum_{i=1}^{5} \gamma_{i1} (x_i-\mu_1^{new})^2$  
$(\sigma _{1}^{new})^{2}\approx \frac{1}{2.5}[0.9821(1-1.924)^{2}+0.8807(2-1.924)^{2}+0.5(3-1.924)^{2}+0.1193(4-1.924)^{2}+0.01179(5-1.924)^{2}])$  
$\sigma_1^{new}= 0.9178$
----------------------------------------------------------------

:::warning[**Be Advised!**]
Boiling water can be dangerous!
:::

## Materials

You need three things to boil water:

::::div{.grid .grid-cols-3 .gap-x-1}

:::div{.col-span-1 .text-center}
A working stove
![stove]
:::

:::div{.col-span-1 .text-center}
A pot
![pot]
:::

:::div{.col-span-1 .text-center}
Water
![water]
:::

::::

## Steps

### Step 1: Put the pot on the stove
Place the pot :icon[cooking-pot] on the stove.

### Step 2: Fill the pot with water
Put some water in the pot.

### Step 3: Turn the stove on
Turn the stove on by turning the knob. :icon[rotate-ccw]{.stroke-orange-500}

:::tip
If you're not sure whether the stove is on, :span[touch the burner]{.underline}. If it's not hot, it means you turned the wrong knob or your stove doesn't work.
:::

### Step 4: Play the waiting game
After about 10 minutes, the water should start to boil. If not, give it more time. Run some errands and when you come back, the water should be boiling 🔥


<!-- Image references --->
[stove]: https://maxima.com/img/w9OGpTeMEP8Xk0pHIDm4At1TFwZify8WJ2igFhlpWGw/resize:fit:700:700/aHR0cHM6Ly9tYXhpbWEuY29tL21lZGlhL2NhdGFsb2cvcHJvZHVjdC83LzgvNzg2NW4xODA5MDgxMGUtNjE2NTZmNmNjZmQ0Yy5qcGc_d2lkdGg9NzAwJmhlaWdodD03MDAmc3RvcmU9ZW4maW1hZ2UtdHlwZT1pbWFnZQ.jpg?type=catalog
[pot]: https://www.ikea.com/us/en/images/products/hemkomst-pot-with-lid-stainless-steel-glass__1083743_pe859078_s5.jpg?f=s
[water]: https://i.natgeofe.com/n/c199c917-f357-45b2-9940-77784ed98d2d/why-is-america-running-out-of-water_16x9.jpg