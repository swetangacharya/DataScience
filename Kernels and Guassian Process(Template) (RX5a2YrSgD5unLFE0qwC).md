---
title: Kernels and Guassian Process(Template)
description: A super secret template post!
is_draft: true
restrict_access: 
  - alice@hotmail.com
  - bob@gmail.com
  - vtrLuoHOjfqb3Er9Sidy3P6VxulB
---

:::info
**New here?**  
Every post begins with a random template to help you start writing.

**Still confused?** 🤔  
- [:icon[video] Watch the **Getting Started** video :icon[external-link]](https://www.youtube.com/watch?v=0h4gRvgoRn4&list=PL9Zhnnyw1lVND99JOWpTyYly9heBP4-Xh){target=_blank}
- [:icon[file-text] Read the **Getting Started** guide](https://www.scipress.io/post/l7R0XuDTe6R1dC2dS5cc/Getting-Started)
:::

---

### Proprietary algorithm

```
initialize foo equal to 1
initialize baz equal to ""

for i = 1, 2, ... 10:
  set baz := baz + "Oo"
```

*******------------------------------------------------------  
#### Kernels:-  
Let us consider a similarity measure of form  
$k:X   X \to \R$  
$(x,x')\mapsto k(x,x')$    
that is, a function that, given two patterns x and x', returns a real number characterzing their similarity. Unless stated otherwise, we will assume that is k is symmetric, that is, $k(x,x')=k(x',x)$ for all $x,x' \in X$.  
- dot product is one way to measure similarity.however, that the dot product approach is not really sufficiently general to deal with many interesting problems.  
-  First, we have deliberately not made the assumption that the patterns actually exist in a dot product space. So far, they could be any kind of object. In order to be able to use a dot product as a similarity measure, we therefore first need torepresent the patterns as vectors in some dot product space $H$ (which need not coincide with $\R^N$ ). To this end, we use a map  
$\phi: X \to H$  
$x \mapsto X:= \phi(x)$                           ...(1.5)  
- second, even if the original patterns exist in a dot product space, we may still
want to consider more general similarity measures obtained by applying a map
(1.5). In that case, $\phi$ will typically be a nonlinear map.  
**Feature Space:- (Learning with kernels.pdf)**  In both the above cases, the space $H$ is called a feature space. Note that we have used a bold face $X$ to denote the vectorial representation of $x$ in the feature space.  
- To summarize, embedding the data into $H$ via $\phi$ has three benefits:  
1. It lets us define a similarity measure from the dot product in  
$\color{purple}k(x,x'):=<x,x'>=<\phi(x),\phi(x')>$  
2.  It allows us to deal with the patterns geometrically, and thus lets us study learning algorithms using linear algebra and analytic geometry.   
3. The freedom to choose the mapping $\phi$ will enable us to design a large variety of similarity measures and learning algorithms. This also applies to the situation where the inputs $x_i$ already exist in a dot product space. In that case, we might directly use the dot product as a similarity measure. However, nothing prevents us from first applying a possibly nonlinear map $\phi$ to change the representation into one that is more suitable for a given problem.     
![Alt text](classification1.png)



**Fig1.1:-** A simple geometric classification algorithm: given two classes of points (depicted by 'o' and '+'), compute their means $c_+$, $c_-$ and assign a test pattern x to the one whose mean is closer. This can be done by looking at the dot product between x — c (where $c = (c_+ + c_-)/2)$ and $w:— c_+ — c_-$, which changes sign as the enclosed angle passes through
$\pi/2$. Note that the corresponding decision boundary is a hyperplane (the dotted line) orthogonal to w.

we begin by computing the means of the two classes in features spaces;  
$\color{purple}c_+= \frac{1}{m_+} \sum_{\set{i|y_i=+1}} x_i$  
$\color{purple}c_-= \frac{1}{m_-} \sum_{\set{i|y_i=-1}} x_i$    
- This geometricconstruction can be formulated in terms of the dot product  
 $ < \cdot, \cdot>$  
 Half way between $c_+$ and $c_-$ lies the point $c:= (c_++ c_-)/2$. we compute the class of x by checking whether the vector x-c connecting c to x encloses an angle smaller than $\pi/2$ with vector $w:=c_+-c_-$ conecting class means. This leads to    
 $y= sgn<(x-c),w>  $  
 = sgn<$(x-(c_++c_-)/2),(c_+-c_-)$>  
 = sgn$<<x,c_+>-<x,c_->+b> $   
 where $b:= \frac{1}{2}(||c_-||^2-||c_+||^2)$,  
 now substitute values of $c_-$ and $c_+$   
 y=sgn$(\frac{1}{m_+} \sum_{\set{i|y_i=+1}}< x_i,x>$ - $\frac{1}{m_-} \sum_{\set{i|y_i=-1}} <x_i,x>+b)$   ... (eq 1.10)
   =sgn$(\frac{1}{m_+} \sum_{\set{i|y_i=+1}}k( x_i,x)$ - $\frac{1}{m_-} \sum_{\set{i|y_i=-1}} k(x_i,x)+b)$.    ...(eq 1.11)  
  if we assume that $b=0$(i.e., the class means have the same distance to the origin), and that k can be viewed as a probability density when one of its arguments is fixed. By this we mean that it is positive and has unit integral.  
$\int_x k(x,x')dx=1$ for all $x' \in X$  
In this case, (1.11) takes the form of the so-called Bayes classifier separating the two classes, subject to the assumption that the two classes of patterns were generated by sampling from two probability distributions that are correctly estimated by the Parzen windows.  
$p_+(x):=\frac{1}{m_+} \sum_{\set{i|y_i=+1}} k(x,x_i)$  and   
$p_-(x): = \frac{1}{m_-} \sum_{\set{i|y_i=-1}} k(x,x_i)$
- Given some point x, the label is then simply computed by checking which of
the two values $p_+(x)$ or $p_-(x)$ is larger, which leads directly to eq. 1.11. Note that this decision is the best we can do if we have no prior information about the probabilities of the two classes.  

$\color{purple}y= sgn(\sum_{i=1}^m \alpha_i k(x,x_i)+b)$   ...(eq.. 1.15)
- $\alpha_i$ can be considered a dual representation of the hyperplans's normal vector. Both classifiers are example-based in the sense that the kernels are centered on the training patterns, that is, one of the two arguments of the kernel is always a training pattern. A test point is classified by comparing it to all the training points that appear in (1.15) with a nonzero weight.  
- Now, In the feature space representation, this statement corresponds to saying that
we will study normal vectors w of decision hyperplanes that can be represented
as general linear combinations (i.e., with non-uniform coefficients) of the training
patterns. For instance, we might want to remove the influence of patterns that are
very far away from the decision boundary, either since we expect that they will not
improve the generalization error of the decision function, or since we would like to
reduce the computational cost of evaluating the decision function (cf. (1.11)). The
hyperplane will then only depend on a subset of training patterns called Support
Vectors.


----------------

**Stationary Kernel:-**  
for real valued inputs $X=\Reals^D$, it is common to use stationary kernels, which are the function of form $\color{blue} k(x,x')=k(||x-x'||)$; thus the value only depends on elementwise difference between the inputs. The RBS kernel is a stationary kernel.    
**Euclidean distance:-** is the straight-line distance between two points in space. It assumes all features (dimensions) are uncorrelated and have equal variance. The formula is:  
$ {d_{Euclidean(x,y)}=((x_1-y_1)^2 + (x_2-y_2)^2+...(x_n-y_n)^2 )^{0.5}}$  
**Mahalanobis Distance:-**
adjusts for correlations and different variances among variables. It measures distance by considering the covariance structure of the data: 
$d_{mahalanobis}(x,y)=((x-y)^TS^{-1}(x-y))^{0.5}$  or  
$D_M=((x-\mu)^T \Sigma^{-1}(x-\mu))^{0.5}$  
Univariate Z-score :- $ Z_i =\frac {x_i- \bar x}{s}$, so $D_M$ is multidimentional representation of the one dimensional Z-score.

**Key Differences:-** 
- Euclidean ignores correlations between variables; Mahalanobis incorporates them.
- Besides SVM, Gaussian process also uses kernel. Also known as kriging.
Here, the employed kernel is determined by the covariance matrix of the training data. With the help of this kernel, a stochastic process is defined for which every point evaluation is normally distributed with certain mean and covariance structure. Algorithmically, the minimization of a
least squares loss function together with a weighted regularization term is computed there.


---

:::info[`restrict_access`]{icon="vault"}
Notice how the YAML frontmatter of this post includes

```
restrict_access: 
  - bob@gmail.com
  - alice@hotmail.com
  - nPUHpcOPgnvzbdggJAx4
```

Only you (the post author) plus these three users will have access to this post once it's published (i.e. `is_draft: true`).

To learn more, see our docs on [Access Control ->](https://www.scipress.io/post/sC51z8nreOEP4Cle6D17/Access-Control)
:::