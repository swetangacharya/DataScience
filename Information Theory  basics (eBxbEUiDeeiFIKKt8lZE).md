---
title: Information Theory  basics
description: Template post about Estimation Theory
slug: code-blocks
is_draft: true
tags:
  - python
  - Entropy and Cross Entropy
  - Fisher Information
  - KL Divergence
  - MLE (Maximum Likelihood Estimation)  
  - Cramer-Rao Lower Bound

---

:::info
**Entropy and Cross Entropy**  
While working with TensorFlow, we come across loss='sparse_categorical_crossentropy' while compiling model (it is related to training your model. Actually, your weights need to optimize and this function can optimize them. In a way that your accuracy make increases. )


https://www.britannica.com/science/information-theory/Entropy  
**Entropy** 
- $H(P)=-\sum p_i log(p_i)$
   - It is a mesaure of how uncertain the events are. It is an average amount of Information you get from one sample when drawn from given probability distribution is. It tells you how unpredictable that probability distribution is.
      1. Let us understand it better. Say, we have Random Variable X and it takes the values $x_i, i=1,...,n$. And it is some probability distribution.  
      **self Information**  
      $ I(x_i)=log\dfrac {1}{p(x_i)}$  
      This shows that less likely the event(i.e., less probability) more the information we gain. As an example if police is analyzing the details of Face of a burglar, and they found that he has big wart on his face that is rare and gives more information.
      2. now let us take the Avg of this self information, call it entropy.
        $H(x)= \sum_{i=1}^{n} I(x_i)P(x_i)$
        ![Alt text](Entropy.png)        
        - see above, the graph shows that higher the probability of an occurring event, lower the suprise (Entropy).
        - In essense entropy is a measure of uncertainity, when our uncertainity is reduced we gain information. Thus, receiving an amount of information is equivalent to having exactly the same amount of entropy taken away.
        - **Important Point**- The entropy of a discrete variable depends 
only on the probability distribution of its values. Changing the values of a discrete variable does not change its entropy provided the number of values stays the same. For ex. if dice values are doubled i.e., {2,4,6,8,10,12,14,16}, the probability associated with it (1/8) doesn't change and $H(X)=3    bits$

**Example**  
Say, we've 8-sided die has 3 bits of entropy. Possible outcomes of Random Variable X =[000,001,010,011,100,101,110,111], now if we have the information about first 2 bits and say they are '01' then our entropy drops by factor of $2^H=2^2=4$,  so our uncertainty about the 
value of a variable is one quarter as big as it was before receiving these 
two bits. Because the die has eight sides, this would mean that we 
now know the outcome is one of only two possible values. (010 or 011), so, out of 8 possibilities we are reduced to 2 possibilities. it is a reduction of 3/4 of uncertainity , hence residual is 1/4 (25%). and by knowing one bit it drops to 50%. So, formula for our residual uncertainity U is $U= 2^{-H}$
, now, half-bit entropy  $2^{-1/2}=0.707= 71\%$, i.e., uncertainity is now reduced to 71% from 100%.  

Supporse we have 2.3bits of Entropy: it means that, on average, each outcome of a random variable or each symbol in a sequence contains 2.3 bits of information or uncertainty. **In practical terms, this value quantifies the minimum average number of binary (yes/no) questions needed to uniquely identify an outcome from the distribution**.

**Maximum Entropy**  
Now, How can We maximize the information transmission. well, the uniform distribution, having Maximum Entropy has Maximum information in every transmitted symbols. which in turn requires Maximum mutual information between input Random Variable X and output Random Variable Y, that requires maximizing entropy.



**Mutual Information**
  - Mutual information is a general measure of association between 
two variables, like the input and output of a communication channel. Mutual information (MI) is defined as a measure of the amount of information that one random variable contains about another random variable.   It has many properties that apply to both discrete and continuous variables.  
- Given two variables X and Y, the mutual information I(X,Y), between them is the average information that we gain about Y after we have observed a single value of X. Because mutual information is a symmetric quantity (i.e., I(X,Y)=I(Y,X)), it is also the average information that we gain about X after we have observed a single value of Y. Equivalently, mutual information is the average reduction in uncertainty about X that results from knowing the value of Y, and vice versa.
- The mutual information between two variables 
X and Y is the average reduction in uncertainty about the 
value of X provided by the value of Y, and vice versa. It is the average amount of information that x conveys about y.  
- **Difference between Mutual Information and Conditional Entropy**    
  * Conditional Entropy measures average uncertainity remained about random variable Y after knowing the value of X.
  *  It tells you how much "noise" or unpredictability is left in Y once X is known. if knowing X completely determines Y then $H(Y|X)=0$. if knowing X doesn't give any info about Y then $H(Y|X)=H(Y).$  
  **$$ H(Y|X)= -\sum_{x \in X} \sum_{y \in Y} (P(x,y)  log_2(P(x|y)))$$**  
  * Conditional entropy is not symmetric, while mutual information is symmetric.
  if X and Y are independent then **$I(X;Y)=0$**.  if they are perfectly dependent then  
  **$I(X;Y)=H(X)=H(Y)$**  
  * in summary, Conditional entropy quantifies the uncertainty left in one variable after observing another, while mutual information quantifies the amount of uncertainty reduced (or information gained) about one variable by knowing the other.  
  ![Alt text](mutual_information.png)

- **Difference between Joint Entropy and Mutual Information**
    - **Joint Entropy** quantifies the total uncertainty associated with two random variables X and Y and defines as:  

      **$$ H(X,Y)= -\sum_{x \in X} \sum_{y \in Y} (P(x,y)  log(P(x,y)))$$**
    - This measure reflects the average amount of information contained in the joint distribution of X and Y.
    - **Mutual Information** measures the amount of information that one random variable contains about another. It quantifies the reduction in uncertainty about one variable given knowledge of the other. It is defined as:  

      **$$I(X;Y)=H(X)+H(Y)-H(X,Y)$$**
      ![Alt text](mutual_information2.png)  
      * in a communication channel if input is $X$ and output $ Y=X+\eta$ where $\eta$ is a noise added during the transmission of input symbols.
      then the Conditional entropy $H(Y|X)$ is the entropy of the channel noise $H(\eta)$ added to the input $X$. from above diagram we can see..  
      > $H(\eta)=H(Y)-I(X,Y) i.e., H(Y|X)=H(\eta)$   
    
    - below python code will be helpful to know the usage of mutual information.
    https://github.com/swetangacharya/DataScience/blob/main/Mutual_Info_Channel_noise.py

    - See below link of python (sklearn) program that shows the usage of Mutual Information in feature selection.
    https://github.com/swetangacharya/DataScience/blob/main/Mutual_Information.py

  -  now in supervise learning we've two probability distribution, one that is predicted by model (Q) and another one True probability distribution (P).  
      >    $$H(P,Q)=-\sum (P \log_e(Q)) $$   
          This formula essentially calculates the expected number of bits needed to encode events from the true distribution P using a coding scheme optimized for the estimated distribution Q. 

      - suppose we have 4 different probable output predicted by model,  where P is a True PDF and Q is PDF calculated by model.
      ```
      import numpy as np
      P=np.array([0.6,0.2,0.15,0.05])
      Q=np.array([0.3,0.4,0.2,0.1])
      entropy=-np.sum(P*np.log(P))
      entropy
      np.float64(1.0627375681569962)
      cross_entropy=-np.sum(P*np.log(Q))
      cross_entropy
      np.float64(1.2621867704852099)
      KL_Divergence=cross_entropy-entropy
      KL_Divergence
      np.float64(0.19944920232821373)

      from scipy.special import rel_entr
      np.sum(rel_entr(P,Q))
      np.float64(0.19944920232821373)
      np.sum(rel_entr(Q,P))
      np.float64(0.19616585060234526)
       

      ```
     > Cross-Entropy:-   if two distribution $p ,  q $,   become similar then cross-entropy is reduced. It is always larger than the Entropy. it is the total entropy after adding $q(x)$ into the system, so extra entropy from   
     $q(x) = H(p,q)-H(p)$, name for this extra entropy is KL Divergence.  
     $D_{KL}=(p||q) =H(p,q)-H(p,p)$

      > KL Divergence= Cross Entropy - Entropy  

      > $D_{KL}(P||Q)=-\sum (P \log_e(P)) -(- \sum (P \log_e(Q)))= 
      \sum(P \log_e(\dfrac {P}  {Q}))$ 

      >$H(P,Q)= H(P)+D_{KL}(P||Q)$ ,  
      where H(P) is the entropy of True distribution P.
      $ D_{KL}(P||Q) $ , quantifies how much additional information is needed when using Q instead of P.
    
    **---Application in Machine Learning---**  
      In machine learning, particularly in classification tasks, cross-entropy is often used as a loss function. For example, when training models like neural networks, minimizing cross-entropy helps improve the model's accuracy by adjusting its predictions to be closer to the true labels. The loss function for a single instance can be expressed as  
      $ L(y,t) = -\sum(t_i \ ln(y_i))$  , where  
       $y_i= pridected \ probability, t_i= True \ class \ labels.$
        

:::
---

#### Fisher Information:-  (https://www.youtube.com/watch?v=pneluWj-U-o)
- Fisher information is a way to measure how much information a random variable or sample provides about an unknown parameter of a probability distribution. It quantifies how sensitive the distribution is to changes in the parameter, and thus how precisely the parameter can be estimated. A higher Fisher information indicates a more precise estimation of the parameter.
- Fisher information focuses on the amount of information that a random variable, or a sample of random variables, carries about an unknown parameter of a probability distribution.  
- It measures how much the probability distribution changes when the parameter is altered. 
- It is directly related to the precision with which the parameter can be estimated. A higher Fisher information implies a more precise estimate.  
$ \color{yellow}I(\theta)= E_\theta \lbrack - \frac{\partial^2}{\partial \theta^2} ln \  p_x(x;\theta) \rbrack$ , where $\theta $ is parameter like mean etc.  or  
$ \color{yellow}I(\theta)= E_\theta \lbrack - \frac{\partial^2}{\partial \theta^2} \ ln \  (\theta;x) \rbrack$  
- what is score:- The score is the derivative (or gradient, for multiple parameters) of the log-likelihood function with respect to the parameter of interest. Mathematically, for a parameter $\theta$ and likelihood function $f(X;\theta)$. For vector parameters, it is the gradient with respect to $\theta$.
- $ \color{red} Score=\frac {\partial}{\partial \theta}log f(X;\theta)$  
- Fisher information is defined as variance of score function.  
$ \color{red} I_X(\theta)=Var \lparen \frac {\partial}{\partial \theta}log f(X;\theta) \rparen$  
-  or, equivalently, the expected value of the squared score:  
$\color{red} I_X(\theta)=E \lbrack \lparen \frac {\partial}{\partial \theta}log \ f(X;\theta) \rparen^2 \rbrack$  
- The score measures how sensitive the log-likelihood is to changes in the parameter $\theta $ indicating the direction and rate at which the likelihood increases or decreases as the parameter changes.
- Covariance Matrix is the inverse of Fisher matrix.  

 **Cramer-Rao Lower Bound (CRLB)**   
 >(https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood)  
 (https://www.numberanalytics.com/blog/ultimate-cramer-rao-bound-guide)  
 
 For example, we've a  sample of independent observations $X=(x_1,x_2,...,x_n)$ from a normal distribution, whose variance $\sigma^2$ is known and mean $\mu$ is unknown. we want to estimate $\mu$.  
 - Likelihood function:- The likelihood function (the probability of observing the data given a particular value of $\mu$  
 $ \color{green} L(\mu,\sigma^2|x_1,x_2,...,x_n)= (2 \pi \sigma^2)^{-n/2} * \exp(\frac{-1}{2 \sigma^2} \sum_{j=1}^n (x_j-\mu)^2)  $  
 - score function:- is the derivative of of the likelihood function with respect to $\mu$.  
 $ \color{green}S(\mu)=\frac{\sum(X_i-\mu)}{\sigma^2}$  
 - Fisher Information:- The Fisher information $I(\mu)$ is the expected value of the square of the score function.  
 $ \color{green}I(\mu)=E[S(\mu)^2]=n/\sigma^2$  
 - Cramer-Rao Lower Bound:-  
 $\color{green}CRLB(\mu)=1/I(\mu)= \sigma^2/n$  
 This means that no unbiased estimator of $\mu$ can have a variance smaller than $\frac{\sigma^2}{n}$ 
CRLB tells you the smallest possible variance you can expect from an unbiased estimatpr for a given situation,allowing you to evaluate how well your chosen estimator performs relative to this theoretical minimum. 

print(FIM)
```
#----output is 3x3 Fisher Information matrix---
#[[10.03908079 -3.15867196  1.43274756]
# [-3.15867196 18.39063302 -0.29376067]
# [ 1.43274756 -0.29376067 18.20723775]]  
Diagonal entries (e.g., 10.04, 18.39, 18.21): These represent the amount of information the data provides about each individual parameter (i.e., the precision of estimating each parameter by itself).
Off-diagonal entries (e.g., -3.16, 1.43, -0.29): These represent the correlation of information between pairs of parameters. If off-diagonal elements are nonzero, it means that the parameters are not estimated independently; changes in one parameter affect the information about another

print(np.linalg.inv(FIM))
## below is the CRLB for each of 3 variables..
#array([[ 0.10647819,  0.01815894, -0.0080859 ],
#       [ 0.01815894,  0.05748637, -0.00050145],
#       [-0.0080859 , -0.00050145,  0.05555141]])  
(CRLB) provides the minimum possible covariance matrix for any unbiased estimator of your parameters. For a vector of parameters, the CRLB is given by the inverse of the Fisher Information Matrix (FIM)  
Diagonal entries: Minimum possible variances for each parameter estimate.  
Off-diagonal entries: Minimum possible covariances between parameter estimates.  
```  
##### Example to calculate Fisher Information for logistic regression:-  
- In logistic regression, the weight matrix is a key parameter that determines how input features are combined to predict the probability of each class. 
- In logistic regression, the "weights" (also called coefficients or parameters) are the values learned for each feature during model training. These weights determine the influence of each feature on the predicted probability of the target class.   
**Interpreting the weights-**  
- Each weight represents the change in the log odds of the outcome for a one-unit increase in the corresponding feature, holding other features constant.  
- The odds ratio for a feature is $exp(\beta_j), $A positive weight increases the odds, a negative weight decreases them.  
**How to use weights-**  
[1] Fit your logistic regression model to obtain the weights $(\beta_0,\beta_1)$.  
[2] compute $p_i$ for each input $x_i$ using fitted weights.  
[3] calculate $ \color{blue}p_i(1-p_i)$  
[4] Sum up the required terms as shown in the matrix above to construct the Fisher Information Matrix.  

| The Fisher Information Matrix is used to estimate the variance-covariance matrix of the weights (by inverting the Fisher Information Matrix).  
|  This allows you to compute standard errors, confidence intervals, and perform hypothesis tests for your logistic regression coefficients.  
$\color{red}p_i= \frac{exp(\beta_0+\beta_1x_i)}{1+exp(\beta_0+\beta_1x_i)}$ ....Fisher_eq1.   
$ \color{blue} I(\theta)=
\begin{bmatrix}
   \sum_{i=1}^n p_i(1-p_i) & \sum_{i=1}^n p_i(1-p_i)x_i \\
   \sum_{i=1}^n p_i(1-p_i)x_i & \sum_{i=1}^n p_i(1-p_i)(x_i)^2
\end{bmatrix} $    ....Fisher_eq2

 we have log-odds is, $ \color{red}log \frac{p_i}{1-p_i}=\beta_0+\beta_1x_{i1}+...+\beta_px_{ip}$ ....Fisher_eq3  
 - ways to find Fisher Information. $I(\theta)= X^TWX$, how?  
 [1] $l(\beta)= \sum_{i=1}^n[y_ilogp_i+(1-y_i)log(1-p_i)]$    
 [2] first derivative is $\sum x_i(y_i-np_i)$   
 [3] the second derivative(Hessian) of the log-likelihood with respect to $\beta$..  
  $\frac{\partial^2l(\beta)}{\partial\beta \partial \beta^T}=\sum [-n*p_i*(1-p_i)*x_ix_i^T]= -X^TWX$  
  Vector form (xTWx): The Hessian can be expressed in a vector form by recognizing that H is a sum of terms like -n * pᵢ * (1 - pᵢ) * xᵢxᵢᵀ. Let W be a diagonal matrix with elements $W_i = -n * p_i * (1 - p_i)$. Then:   
   $\color{green}H=\sum W_i*x_ix_i^T=X^TWX$, W is a diagonal matrix with elements related to the variance of the predicted probabilities. 
## Here's a code block
Scipress code blocks make your code look beautiful!

```python 
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
np.set_printoptions(suppress=True,precision=4)
from pprint import pprint

def fun1(inputs,weights,intercept):
    '''this is find the value of pi=math.exp(weights[0]*xi)/(1+math.exp(weights[0]*xi'''
    b1=weights[0]
    log_odds=intercept+b1*X.flatten()
    odds=np.exp(log_odds)
    p_i=odds/(1+odds)
    I11=np.sum(p_i*(1-p_i))
    I12=np.sum(inputs*p_i*(1-p_i))
    I21=I12
    I22=np.sum(inputs*inputs*p_i*(1-p_i))
    mat=[[I11,I12],[I21,I22]]
    pprint(mat)
    
x = np.array([25.0, 26.5, 28.2, 29.2, 31.4, 32.5, 34.6, 35.7, 37.8, 38.5, 40.1, 41.6, 43.1, 44.6, 46.1, 47.6, 49.1, 50.6, 52.1, 53.6, 55.2, 56.7, 58.2, 59.7, 61.2, 62.7, 64.2, 65.7, 67.2, 68.7, 70.3, 71.8, 73.3, 74.8, 76.3, 77.8, 79.3, 80.8, 82.3, 83.8, 85.4, 86.9, 88.4, 89.9, 91.4, 92.9, 94.4, 95.9, 97.4, 99.0])
y = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

# Reshape x for sklearn
X = x.reshape(-1, 1)

# Fit logistic regression
model = LogisticRegression(solver='lbfgs')
model.fit(X, y)

intercept=model.intercept_[0]
weights= model.coef_
print('weights are', weights)
print('intercept is= ', intercept)
fun1(x,weights,intercept)

p = model.predict_proba(X)[:,1]
V=np.diag(p*(1-p))

#add intercept to X
X_design = np.hstack([np.ones((X.shape[0], 1)), X])
FIM=X_design.T @ V @ X_design
pprint(np.round(FIM))
pprint(FIM)
```  
- **print intercept and weights**  
weights are [[1.0463]]  
intercept is=  -58.52290111749495    

**- output of the fun1 from above program. (from Fisher_eq2 above)**  

[[0.6280050394680543, 35.142209408067686],  
 [35.142209408067686, 1968.396250925736]]

**- output of last two line of above code (i.e., $X^T W X$)**  
array([[   1.,   35.],  
       [  35., 1968.]])  
array([[   0.628 ,   35.1422],  
       [  35.1422, 1968.3963]])

:::info[Want to learn more?]
See our docs on [Code ->](https://www.scipress.io/post/jKgNqkgB1k6saoDUKkiP/Code)  
[docs ->]  
https://christophm.github.io/interpretable-ml-book/logistic.html  
https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture26.pdf  
https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf
:::
