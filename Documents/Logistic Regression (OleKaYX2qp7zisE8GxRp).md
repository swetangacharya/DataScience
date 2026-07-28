---
title: Logistic Regression
description: Template post about dogs; They're like humans, but better.
slug: why-dogs-are-amazing
is_draft: true
icon: cat
tags:
  - Bernoulli distribution
  - Regression
  
---

:::info




**Still confused?** 🤔  
- [:icon[video] Watch the **Getting Started** video :icon[external-link]](https://www.youtube.com/watch?v=0h4gRvgoRn4&list=PL9Zhnnyw1lVND99JOWpTyYly9heBP4-Xh){target=_blank}
- [:icon[file-text] Read the **Getting Started** guide](https://www.scipress.io/post/l7R0XuDTe6R1dC2dS5cc/Getting-Started)
:::







## What is a Logistic Regression? 
- Essentially, when we talk about the logit function, we are working with the function of a random variable p, more specifically, one corresponding with a Bernoulli distribution.
- A random variable takes value 1 with success probability p and 0 with failure probability q = 1-p.  
-  The logit function is a quantile function associated with the standard logistic distribution. It has many uses in data analysis and machine learning, especially in data transformations.  
- Mathematically, the logit function is inverse of standard logistic function (sigmoid function. 
    $\sigma(x)=  \frac {1}{1+e^{-x}}$ ,  so the $\sigma^{-1}(p)= ln \frac {p}{1-p}$  
    ![Alt text](logit_function.png)  Fig.1
    - in Fig. 1 the blue curve represents $\frac {1}{1+e^{-x}}$, the logistic function, and red curve represents logit function  $ln \frac {p}{1-p}$.  
    ![Alt text](sigmoid.png).  Fig. 2  
    - Binary logistic regression is a type of regression analysis that is used to estimate the relationship between a dichotomous dependent variable and dichotomous-, interval, and ratio-level independent variables.
    - dichotomous variables (like Bernoulli's RV), are coded as '1' for success or yes or '0' for failure or no.
       - **why/when do we need it?**  
          - In many cases, it is not unreasonable to assume that the relationship between two variables is non-linear.
          - Let us think about a scenario when a person having salary \$40k many not buy own house. ,but increase of \$10k he may buy own house.
          - now if person has salary \$10k with increase of \$10k may not buy house, and on the same note, of a rich person has salary \$90k may not have impact of adding extra \$10k, as he already passed the threshould. that is where the logistic function helps.  
          #### problem with Ordinary Least Square(OLS) Regression  
            -The error term is not normally distributed when you use OLS regression with a dichotomous dependent variable because, for any value of X, there are only two possible values that the residuals can take. A residual is defined as the observed value on the dependent variable minus the predicted value given X.  
            - For ex. our regression function is $ \hat y_i = 0.03 + 0.48x_i$, now our R.V can take only two outputs $y_1=1 $and $y_0=0$, now for RV X=2, we have residual $ \hat y - y$  
            $1- \hat y= 1- (0.03 + 0.48*2)=0.01$  
            $0- \hat y= 0- (0.03 + 0.48*2)=-0.99$
            - Thus, for any value of x, there are only two possible residuals so the distribution is not normal.
            -Another issue is of Heteroskedasticity, If the variance of the errors is not constant across observations, OLS estimators can still be unbiased, but they are no longer efficient (i.e., they do not have minimum variance). this violates the BLUE (Best Linear unbiased estimator), according to Gauss-Marcov assumptions.
            LS estimators should be unbiased, efficient, and consistance across various samples taken from population.
            - The mean of Bernoulli's distribution is $\mu =p$ and variance is $var= p(1-p)$
            - This simply means there are no other linear unbiased estimator which has lower variance than that is assumed by current estimator. One of the important assumption of Least Square is _:hl[zero conditional mean  assumption]_,   
            Mathematically $ E (\epsilon|X)=0$, i.e., $cov(\epsilon,X)=0$.
            - Another issue is  Heteroskedasticity leads to biased estimates of the standard errors, which we use in our t tests. Poor estimates increase the chance of drawing incorrect conclusions in hypothesis testing.  While heteroskedasticity does not bias the coefficient estimates themselves, it does result in biased estimates of their standard errors. Ordinary Least Squares (OLS) assumes constant variance of errors, and when this assumption is violated, the calculated standard errors may be either overestimated or underestimated.This inconsistency can mislead researchers regarding the statistical significance of their findings.  
            :hl[**When topic of interest are dichotomous!!**]  
            - Logistic regression uses the logit transformation to linearize the non-linear relationship between X and
              the probability of Y. It does this through the use of odds and logarithms. So, the logit is a nonlinear function thatrepresents the s-shaped curve.  
                $logit(p)= ln(\frac{p}{1-p})=\beta_0 +\beta_1x_1+\beta_2x_2$ , then ,
                odds= $\frac {p_i} {1-p_i}= e^{\beta_0}*e^{\beta_1 x_i} *e^{\beta_2 x_i}$  
                probabilities = $P_i=E(Y_i=1|X_{1i},X_{2i}) = \frac {1}{1+e^{-(\beta_0 +\beta_1x_i+\beta_2x_i)}}=\frac {e^{(\beta_0 +\beta_1x_i+\beta_2x_i)}}{1+e^{(\beta_0 +\beta_1x_i+\beta_2x_i)}} $
            - Now, one may ask, how does it work, from where the value of $\beta$ comes? Logistic regression uses     maximum likelihood estimation to generate estimates of $\beta$. Specifically,ML uses the observed data and probability theory to find the most likely or the most probable 
            population value given the sample data(observations). In logistic regression, the formula that is used to determine the population value most likely to yield the sample data is given by the likelihood function.  
            $LF = \prod  \left\{P_i^ {y_i} * (1-P_i)^{1-y_i} \right\}$  
            -The likelihood function is an expression for the likelihood of observing the pattern of 
            occurrences (y=1) and non-occurrences (y=0) of an event in a given sample. In other words, it 
            tells us the probability of getting our sample data from a population with probabilities equal to $P_i$

- Those pairs $(x_i,y_i)$, where $y_i=1$, the contribution to the likelihood function is $p(x_i)$, and for those
pairs where $y_i=0$, the contribution to the likelihood function is $1-p(x_i)$, where $p(x_i)$ denotes the value of $p(x)$ computed at $x_i$.
A convenient way to express the contribution to likelihood function for pair $(x_i,y_i)$ is through expression  
$\color{orange}p(x_i)^{y_i}[1-p(x_i)]^{1-y_i}$  
- Now from above Likelihood function (LF) converted as log-likelihood   
$\color{orange}LF(\beta)=ln[LF(\beta)]= \sum_{i=1}^n{y_i \ln[p(x_i)]+(1-y_i)ln[1-p(x_i)]}$

### Understanding Logistic Regression  
**- Interpretation of Weights:-**  
The interpretation of the weights in logistic regression differs from the interpretation of the weights in linear regression since the outcome in logistic regression is a value between 0 and 1. The weights don’t influence the probability linearly any longer. To interpret the weights, we need to reformulate the equation for the interpretation so that only the linear term is on the right side of the formula.  
$\color{orange}ln (\frac{P(Y=1)}{1-P(Y=1)})=ln (\frac{P(Y=1)}{P(Y=0)})=\beta_0+\beta_1x_1+...+\beta_px_p $  
> On the level of probabilities, logistic regression is not linear in the features. Meaning an increase by one unit in the features doesn’t increase the probability by , but rather changes the probability multiplicatively. 

$\color{orange}\frac{odds_{x_j+1}}{odds_{x_j}} =\frac{exp(\beta_0+\beta_1x_1+...+\beta_j(x_j+1)+...+\beta+px_p)}{exp(\beta_0+\beta_1x_1+...+\beta_jx_j+...+\beta+px_p)}=exp(\beta_j(x_j+1)-\beta_jx_j)=exp(\beta_j)$  
- In the end, we have something as simple as exp()
 of a feature weight. A change in a feature by one unit changes the odds ratio (multiplicative) by a factor of $exp(\beta_j)$  
 - A change in $x_j^{(i)}$ by one unit increases the log odds ratio by the value of the corresponding weight.
- if you have odds of 2, it means that the probability for $Y=1$ is twice as high as $Y=0$. if you have a weight(log odds ratio) of 0.7, then increasing the respective feature by one unit multiplies the odds by $exp(0.7)=e^{0.7}=2.07$, and the odds change to 4.  
- Logistic regression can also be extended from binary classification to multi-class classification.  
- disadvantage of the logistic regression model is that the interpretation is more difficult because the interpretation of the weights is multiplicative and not additive.  
- Logistic regression can suffer from complete separation. If there is a feature that would perfectly separate the two classes, the logistic regression model can no longer be trained. This is because the weight for that feature would not converge, because the optimal weight would be infinite. This is really a bit unfortunate because such a feature is really useful. But you do not need machine learning if you have a simple rule that separates both classes. The problem of complete separation can be solved by introducing penalization of the weights or defining a prior probability distribution of weights.   
-  #### significance of coefficients:-   
- After estimating the coefficients, our first look at the fitted model commonly concerns an assessment of the significance of the variables in the model. This usually involves formulation and testing of a statistical hypothesis to determine whether the independent variables in the model are “significantly” related to the outcome variable.  
- **Does the model that includes the variable in question tell us more about the outcome (or response) variable than a model
that does not include that variable?**  
- If the predicted values with the variable in the model are better, or
more accurate in some sense, than when the variable is not in the model, then we feel that the variable in question is “significant” .  
$\color{red}D=-2ln [\frac {likelihood of the fitted model}{likelihood of the saturated model}]$

$\color{red}D=-2 \sum_{i=1}^n[y_i ln (\frac{\hat p(x_i)}{y_i})+(1-y_i)ln(\frac{1-\hat p(x_i)}{(1-y_i)})]$ , this is called **deviance.**
- $\color{red}l(saturated model)= \prod_{i=1}^n y_i^{y_i}x (1-y_i)^{(1-y_i)}=1.0$



----------
The typical setup for logistic regression is as follows: there is an outcome 
y that falls into one of two categories (say 0 or 1), and the following equation is used to estimate the probability that 
y belongs to a particular category given inputs $(X=(x_1,x_2,...,x_k)$   
:::info
**$P(y=1|X)=sigmoid(z)= \frac {1}{1+e^{-z}} $    where $z= \hat \beta_0+ \hat \beta_1 x_1+ ...+\hat \beta_kx_k$**  
 :::
$z$  is called a linear predictor, and it is transformed by the sigmoid function so that the values fall between 0 and 1, and can therefore be interpreted as probabilities. This resulting probability is then compared to a threshold to predict a class for $y$ based on $x$.   
* we have a model (here linear model with parameters $\hat \beta_0, \hat \beta_1$), then we are trying to find the probability of $P(y=1|X, \theta)$ where $\theta$ is $\hat \beta_0, \hat \beta_1$.  
* now we want to use hypothesis as linear regression $\theta ^T X$, but the output should be probability and its value should be between 0 and 1.  
![Alt text](sigmoid1.png)   

* Advantage of using sigmoid: [1] it gives the output between 0 and 1. [2] derivative of sigmoid is easy to calculate.
:::info  
$P(y=0|X)= 1- sigmoid(z)=1-\frac {1}{1+e^{-z}}= \frac {e^{-z}}{1+e^{-z}} $  
:::  
* Bernoulli trials: beause we have two outcomes, the probability of success is same for each trial and outcomes from different trials are independent, we can use Binomial distribution that is the n number of Bernoulli trials.

#### $b(k;n,p)= {\binom {n}{k}} p^k (1-p)^{n-k}$ where $k=0,1,2,...,n$
i.e, number of success in n trials.

Mathematically, for a single training data point (x, y), logistic regression assumes: 
$P(Y=1|X=x) = \sigma(z)$ where $z=\theta_0+ \sum_{i=1}{m}\theta_i x_i$  
$P(Y=0|X=x) = 1-\sigma(z)$

The output of our logistic regression function is
supposed to be the probability that the label is one. This means that we can (and should) interpret each label as a Bernoulli random variable:  
**$Y~Ber(p)$ where $p=\sigma(\theta ^ Tx))$**
probability of one data point    
**$P(Y=y|X=x)= \sigma(\theta ^ Tx)^y \cdot [1-\sigma(\theta^Tx)]^{1-y}$**  
$L(\theta) = \Pi_{i=0}^{n} P(Y=y^{(i)}|X=x^{(i)}) $  
$L(\theta)=\Pi_{i=0}^{n}  \sigma(\theta ^ Tx^{(i)})^{y(i)} \cdot [1- \sigma(\theta ^ Tx^{(i)})]^{1-y(i)}$

And if you take the log of this function, you get the log likelihood for logistic regression. The log likelihood equation is:  
**$LL(\theta) = \sum_{i=1}^n y^{(i)}log (sigma(\theta^T x^{(i)})) + (1-y^{(i)}log[1-\sigma(\theta^Tx^{(i)})]$**  

The only remaining step is to chose parameters (θ) that maximize log likelihood.






































