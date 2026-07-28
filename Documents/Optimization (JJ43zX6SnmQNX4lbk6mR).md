---
title: Optimization 
description: Template post of a Non-Linear Approximation
slug: Non-Linear-Approximation
is_draft: true
icon: shopping-cart
tags:
  - Function Approximation
  - Taylor's Theorem
  - Non Linear Approximation 
  
---

:::info
**New here?**  
Every post begins with a random template to help you start writing.

**Still confused?** 🤔  
- [:icon[video] Watch the **Getting Started** video :icon[external-link]](https://www.youtube.com/watch?v=0h4gRvgoRn4&list=PL9Zhnnyw1lVND99JOWpTyYly9heBP4-Xh){target=_blank}
- [:icon[file-text] Read the **Getting Started** guide](https://www.scipress.io/post/l7R0XuDTe6R1dC2dS5cc/Getting-Started)
:::

---

<!--- Toggle the "Subscribed" button in the navbar to see how this post looks to unsubscribed readers --->

## Weierstrass' Theorem
  Given a Function $f$ defined on Interval $I$, we ask whether we, for a given $\epsilon>0$, can we find a polinomial P such that  
  $|f(x)-P(x)| \leq \epsilon$ for all $x\in I$, a general for of polynomial is $\sum_{n=0}^N a_nx^n$  
  **Theorem:** Let $I \subset R$ be a closed and bounded interval and $f$ a continuous function defined on $I$. Then for every $\epsilon>0$ there exists a polynomial $P$ such that  
  $|f(x)-P(x)| \leq \epsilon$ for all $x\in I$  
  The geometric meaning of the theorem is that if we surround that graph of $f$ with a band having width $2\epsilon$ for some $\epsilon>0$, then there exists a polynomial that goes completely inside this band, regardless how narrow the band is.  
  The approximating polynomial P will in general depend on at least three factors, namely,  
* how well we want f approximated, i.e., how small $\epsilon$ is;
* the behavior of $f$; strong oscillations in $f$ usually force $P$ to be of a high degree;
* the length of interval $I$; enlarging the interval in general forces us to choose polynomials of higher degree if certain approximation has to be obtained.
* for example, for $\epsilon=1$, we may choose $2^{nd}$ order polynomial and for  $\epsilon=0.1$, we may select $10^{th}$ order polynomial.

## Taylor's Theorem

1. Assuming function $f$ is differentiable at $x_0$, then the tangent line usually approximate function $f$ well in a small neighbourhood of $x_0$; for this reason, **the tangent line is called the approximating polynomial of degree 1.**  
$P_1(x)=f(x_0)+f'(x_0)(x-x_0)$ , now if our actual function is $x^2-2x$, but we're only given that slope at $x_0=3$ is 4. then polynomial will be  
$P_1(x)=f(3)+f'(3)(x-3)= 3+4(x-3)=4x-9$ , so at x=2 we have  
$P_1(2)=4(2)-9=-1$, but actual value is 0, hence we need further approximation, i.e., $2^{nd}$ order Taylor polynomial.  
$P_2(x)= f(x_0)+f'(x_0)(x-x_0)+\frac{f''(x_0)}{2!}(x-x_0)^2$, this in turn requires second derivative of that function at 3, say its second derivative is 2 at point $x_0=3$, this gives better information about the curvature of a function.  
$P_2(x)= 3+(4)(x-3)+\frac{2}{2!}(x-3)^2= x^2-2x$, and this is the perfect approximation of our actual function.
take $P_2(1)=3+(4)(-2)+(-2)^2=-1$, it gives exact match!!  
**How??**  
we have **$P_2(x)= f(x_0)+f'(x_0)(x-x_0)+\frac{f''(x_0)}{2!}(x-x_0)^2$**, now put   
At $x=x_0$ , then $P_2(x_0)=f(x_0)$   ,  
differentiating $\frac {d P_2(x)}{dx}=f'(x_0)+f''(x_0)(x-x_0)$, now at $x=x_0$,    
$\frac {d P_2(x)}{dx}|_{x=0}= f'(x_0)$  , look at the actual function and tangent.
![Alt text](Taylor.png)

now, adding an offset of 1 to our function $f$...  
![Alt text](Taylor2.png)
### Overview of Algorithms  
All
algorithms for unconstrained minimization require the user to supply a starting point,
which we usually denote by $x_0$. The user with knowledge about the application and the
data set may be in a good position to choose $x_0$ to be a reasonable estimate of the solution. Otherwise, the starting point must be chosen by the algorithm, either by a systematic approach or in some arbitrary manner.  
Beginning at $x_0$, optimization algorithms generate a sequence of iterates $\{x_k\}_{k=0}^\infty$, that terminate when either no more progress can be made or when it seems that a solution point has been approximated with sufficient accuracy. In deciding how to move from one iterate xk to the next, the algorithms use information about the function $f$ at $x_k$ , and possibly also information from earlier iterates $x_0,x_1,...,x_{k-1}$. They use this information to find new iterate $x_{k+1}$ with a lower function value than $x_k$. ) < f (xk−m).)
There are two fundamental strategies for moving from the current point $x_k$ to a new iterate $x_{k+1}$.  
**Line Search and Trust Region**


##### Adam Optimizer  
It's an adaptive learning rate method that combines the benefits of both momentum and RMSprop, making it efficient and robust for a wide range of tasks. Adam stands for Adaptive Moment Estimation, and its core idea is to adjust the learning rate for each parameter individually based on the history of past gradients.   
**Gradient Descent and Partial Derivative.**  
- Gradient descent is an optimization algorithm that updates the model parameters $w4 to minimize the loss function.  
- To do this, we compute the gradient (vector of partial derivatives) of the loss function with respect to each parameter $w_i$.  
- the partial derivative  $\frac {\partial L}{\partial w_i}$ , tells us how much the loss would change if we make a small change $w_i$  

- update rule for each $w_i$ is  $w_{i(new)} \gets w_{i(old)} - \eta \frac {\partial L}{\partial w_i}$, $\eta$ is a learning rate.  
- By computing $\frac {\partial L}{\partial w_i}$, we find the direction in which to adjust $w_i$ to reduce the loss.  
- This process is repeated for all weights and over many iterations, gradually leading the model to a set of parameters that minimizes the loss on the training data.  
- The loss function $L(y_i,x_i|w)$ quantifies model error for a given data point. Gradient descent uses the partial derivative of this loss with respect to each weight $w_i$ to determine how to adjust the weights to minimize the overall error.  
- Gradient descent is known as a very efficient local optimizer. It can
often be observed that such an algorithm leads to a steep decline of the loss
values before learning seems to slow down. One problem with the
algorithm is that it can, strictly speaking, only find local minima.   
- An analogy would be to think about a ball rolling
downhill on the loss surface. With the basic gradient descent we are always
strictly going downhill. However, with a real ball we would have
momentum so that the ball could overcome a small hill if the momentum is
great enough. To incorporate momentum into the gradient descent
algorithm, we can modify the update so that we take some percentage of the
previous step into account.   
$\color{yellow}\Delta w(t)= - \alpha \nabla L(t) + m \Delta w(t-1)$   
![Alt text](Adam.png)  
- A momentum term of m = 0.9 is a common starting value, but such
hyperparameters of the algorithms are of course problem-dependent and
need to be evaluated on a case-by-case basis.
Local minima have often been stated as a main difficulty for gradient
descent, although true local minima are increasingly difficult to realize in
higher dimensions. To be a true local minimum, it is necessary that all
changes and all combination of changes of all directions of the parameters
lead to larger loss values. It is clear that with increasing dimensions there is therefore an increasing chance to find an “escape route”. However, even so, it is now known that true local minima are not likely to be the problem with most high-dimensional learning scenarios, saddle points, or at least shallow areas of the loss functions seem to present a problem in many applications. A momentum term is a common way to help with such shallow areas.  
- There are additional techniques in common use today. For example, we
can change the learning rate based on the history of the learning
performance, and such adaptive learning rates are now commonly used. For
example, a very popular algorithm is the **ADAM optimizer**. **ADAM stands
for adaptive moment estimation** which is a slight modification of the
momentum method. Instead of strictly using the last entry of the gradient as
the momentum, the ADAM method uses a sliding average of the gradient
$ \color{yellow} m \gets \alpha_1m + (1-\alpha_1) \nabla L^{(i)}$ ,  
and variance of a gradient $ \color{yellow}v \gets \alpha_2v + (1-\alpha_2) (\nabla L^{(i)})^2$    
to modulate the update with the gradient. The model parameters are thereby
updated according to  
$\color{yellow} w \gets w-\alpha \frac {m/(1-\alpha_1)}{(v/(1-\alpha_2))^{0.5}+\epsilon}$,  where the small factor $\epsilon$ is added to prevent possible divisions by $0$.  

##### Constrained Optimization:- 
The goal of constrained optimization is to **identify the local extreme** values of a function
$f(x, y)$ on some restricted (or constrained) domain identified using a curve g$(x, y) = C$  
find the points (x, y) that solve the equation $\color{orange}\nabla f(x,y)= \lambda \nabla g(x,y)$ for some constant $\lambda$ , the number $\lambda$ is called **Lagrange Multiplier**. If there is a constrained maximum
or minimum, then it must be such a point.  
**Example:** For a rectangle whose perimeter is 20 m, use the Lagrange multiplier method to find the dimensions that will maximize the area.  
with x and y representing the width and height,
respectively, of the rectangle, this problem can be stated as:  
 - Maximize : $f(x, y) = xy$ , given:  $g(x, y) = 2x + 2y = 20$ (x+y=10) .
 then $y=(x-10)$  
 now solving equation $\nabla f(x,y)= \nabla g(x,y)$ for some $\lambda$ means solving the equations  
 $\color{orange}\frac{\delta f}{\delta x}= \lambda\frac{\delta g}{\delta x}$  ,  
$\color{orange}\frac{\delta f}{\delta y}= \lambda \frac{\delta g}{\delta y}$  
namely: $y=2\lambda$ and $x=2\lambda$ , The general idea is to solve for $\lambda$ in both equations, then set those expressions equal
(since they both equal $\lambda$) to solve for x and y. Doing this we get $\frac{y}{2} =\lambda= \frac{x}{2}$, $x=y$ 
so now substitute either of the expressions for x or y into the constraint equation to solve for x and y:  
$20=g(x,y)=2x+2y=2x+2x=4x  \implies x=5 \implies y=5$ 



