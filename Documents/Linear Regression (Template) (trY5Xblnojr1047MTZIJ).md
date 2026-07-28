---
title: Linear Regression (Template)
description: Template post about code blocks
slug: code-blocks
is_draft: true
tags:
  - Regression
  - Taylor Theorem
  - pandas
  - linear model
  - ANOVA, F Test
---

:::info
**New here?**  
Every post begins with a random template to help you start writing.

**Still confused?** 🤔  
- [:icon[video] Watch the **Getting Started** video :icon[external-link]](https://www.youtube.com/watch?v=0h4gRvgoRn4&list=PL9Zhnnyw1lVND99JOWpTyYly9heBP4-Xh){target=_blank}
- [:icon[file-text] Read the **Getting Started** guide](https://www.scipress.io/post/l7R0XuDTe6R1dC2dS5cc/Getting-Started)
:::

---

## Generalized Linear Model
  Models may be broadly classified as deterministic or probabilistic. 
  In a deterministic model, the system outcomes and responses are precisely defined, often by a set of equations.
  Deterministic models abound in the sciences and engineering like $PV=nRT$.  
  - In probabilistic models, the system outcomes or responses exhibit variability, because the model either contains random elements or is impacted in some way by random forces, like below...  
      ##### $y=\beta_0+\beta_1x_1+...+ \beta_kx_k$   , here y=outcome or  response variable, $x_0,x_1,...,x_k$ are also called covariates. $\beta_0,\beta_1,...,\beta_k$ are set of unknown parameters.
  
  ##### To use the OLS (Ordinary Least Squares) method to estimate and make inferences about the coefficients in linear regression analysis, a number of assumptions must be satisfied   >>
    1. Measurement: All independent variables are interval, ratio, or dichotomous; the dependent variable is continuous, unbounded, and measured on an interval or ratio scale. All variables are measured without error. 
    2. Specification: All relevant predicators variables of dependent variables are included and no irrelevant predictors of the dependent variable are included in analysis.
    3. Mean/Expected value of error is 0.
    4. Homoscedasticity: The variance of the error term, is the same, or constant, for all values of the independent variables. 
    5. Normality of errors: Errors are normally distributed, for each set of values of independent variables.
    6. No Autocorrelation: No correlation between the error terms.
    7. No correlation between error terms and independent variables.
    8. No multi colinearity between independent variables.
  
  

  **Error in Linear Regression**  
  - When solving linear systems, we seek the single point
that lies on n given lines. In regression, we are instead given n points, and
we seek the line that lies on “all” points.The big difference in defining linear regression is that we seek a line that comes as close as possible to hitting all the points.  
- The residual error of a fitted line $f(x)$ is the difference between the predicted
and actual values. shown in fig.1, for a particular feature vector $x_i$ and corresponding target value $y_i$ , the residual error $r_i$
is defined as $r_i=y_i-f(x_i)$. Least squares regression minimizes the sum of the squares of the residuals of all points.
![Alt text](residula_error.png)  fig.1

- Linear regression seeks the line $y = f(x)$ which minimizes the sum of the squared errors over all the training points, i.e. the coefficient vector $w$ that minimizes  
 #####   $ \sum_{i=1}^{n} (y_i-f(x_i))^2$, where $f(x)= w_0 + \sum_{i}^{m-1} w_ix_i$  
 - Suppose we are trying to fit a set of n points, each of which is m dimensional.
The first m - 1 dimensions of each point is the feature vector $(x_1, . . . , x_{m-1})$,with the last value $y = x_m$ serving as the target or dependent variable. Let us encode this n feature vector in  $n$ x $(m-1)$ maxtix. we can prepend column vector of all ones (1), to make it n x m matrix. This can be think as a constant feature and can be used to get y-intercept. Further n target variables can be made as nx1 column vector b.  
- The optimal regression line $f(x)$ we seek is defined by an m x 1 vector of
coefficients $w = {w_0, w_1, . . . , w_{m-1}}$. Evaluating this function on these points
is exactly the product $A\cdot w$, creating an nx1 vector of target value predictions. Thus $(b - A · w)$ is the vector of residual values.
$w=(A^T A)^{-1}A^Tb$


- Usually,  regression analysis leads to situations where relationship among variables are not deterministic (i.e., not exact), this means there is a random component to the equation that relates
the variables. This random component takes into account considerations that are not being measured or, in fact, are not understood by scientists or engineers. In most cases
$Y=\beta_0 + \beta_1x$, is an approximation that is a simplification of something unknown and complicated.
 Theselinear structures are simple and empirical in nature and are thus called empirical models.
An analysis of the relationship between Y and x requires the statement of a statistical model.

The model must include the set ${(x_i, y_i); i = 1, 2,...,n}$
of data involving n pairs of (x, y) values. One must bear in mind that the value $y_i$ depends on $x_i$ via a linear structure that also has the random component involved. The basis for the use of a statistical model relates to how the random variable $Y$ moves with $x$ and the random component.

The model also includes what is assumed about the statistical properties of the random component. and that is
 
 ###### $Y= \beta_0 + \beta_1x + \epsilon$  
 ![Alt text](True_reg.png) fig.2
- The line in the graph is the true regression line. The points plotted
are actual (y, x) points which are scattered about the line. Each point is on its
own normal distribution with the center of the distribution (i.e., the mean of y) falling on the line. This is certainly expected since $E(Y ) = \beta_0 + \beta_1x$. As a result,the true regression line goes through the means of the response, and the actual observations are on the distribution around the means. Note also that all distributions have the same variance, which we referred to as $\sigma^2$. Of course, the deviation between an individual y and the point on the line will be its individual $\epsilon$ value. This is clear since
$y_i - E(Y_i) = y_i - (\beta_0 + \beta_1x_i)$ = $\epsilon_i$.
Thus, at a given x, y and the corresponding $\epsilon$ both have variance $\sigma^2$.   
![Alt text](comparing_residual1.png)  fig. 3  
 **Fitted Model**
  - we can see from the fig. 3 that $b_0$ is estimate for $\beta_0$ and $b_1$ is the estimate for $\beta_1$ in our predicted model $\hat{y}= b_0+b_1x$. we shall find $b_0$ and $b_1$ such that square of residual minimizes.

  - The residual sum of squares is often called the sum of squares of the errors about the regression line and is denoted by SSE. This minimization procedure for estimating the parameters is called the **method of
least squares**. Hence, we shall find a and b so as to minimize.

  
  **$SSE= \sum_{i=1}^{n} e_{i}^{2}= \sum_{i=1}^{n}(y_i- \hat{y_i})^2 =\sum_{i=1}^{n}(y_i-b_0-b_1x_i)^2$**  
  ![Alt text](SSE1.png)  fig. 4  
  - from fig. 4 we can see that larger the SSR lower the $R^2$, which shows model is not so good. i.e., $R^2$ gives the measure of goodnees of fit.
  
  **Properties of Least square Estimators**  
    $Y_i=\beta_0+\beta_1x+\epsilon_i$, we assume that error term $\epsilon_i$ is random variable with mean 0 and constant variance $\sigma^2$. suppose that we make
the further assumption that $\epsilon_1,\epsilon_2,...\epsilon_n $ are independent from run to run in the
experiment. This provides a foundation for finding the means and variances for the estimators of $\beta_0$ and $\beta_1$.if experiment is repeated over and over again, with fixed values of $x$ everytime then value of $\beta_0$ and $\beta_1$ will most likely to differ from experiment to experiment. These different estimates may be viewed as values assumed by the random variables $B_0$ and $B_1$, while $b_0$ and $b_1$ are specific realizations. Since $x$ is fixed, values of $B_0$ and $B_1$ depends on variations in the values of y, or, more precisely, values of RV $Y_1,Y_2,...Y_n$ . distributional assumptions imply that the $Y_i$, i=1,2,...n, are also independently distributed, with mean $\mu_{Y|x_i}= \beta_0 + \beta_1x_i $ and equal variance $\sigma^2$ . i.e., $\sigma^2_{Y|x_i}=\sigma^2$.   
So, the estimator  $B_1$  is  
    $B_1=\sum_{i=1}^n \frac {(x_i- \bar x)Y_i}{\sum_{i=1}^n (x_i-\bar x)^2} $  
    $\mu B_1=\beta_1$ and  $\sigma^2B_1= \frac {\sigma^2}{(x_i-\bar x)^2}$
- $\mu B_0= \beta_0 $ and $ \sigma^2_{\beta_0}=\sum_{i=1}^n \frac {x_i^2}{n\sum_{i=1}^n (x_i-\bar x)^2 } \cdot \sigma^2$

#### Hetroscendasticity  ...
- Till now we have assumed that all the residuals or error term are generated from the same population that has a constant variance (homoscedasticity). we need these random residuals (no patterns) that are uncorrelated and uniform. Generally, if we see patterns in the residuals, our model has a problem, and we might not be able to trust the results.
This occurs more often in datasets that have a large range between the largest and smallest observed values. While there are numerous reasons why heteroscedasticity can exist, a common explanation is that the error variance changes proportionally with a factor. This factor might be a variable in the model.  
- This variability can cause different variance across different samples and though $\beta_1$ is fixed the variance of the curve varies. Due to this Hypthesis test, test of significance becomes unreliable. 
when can Hetroscendasticity occur?  
1.  cross-sectional data- it often have large variations in data. For ex. income of  USA population and income in other underdeveloped country differes much.  
2. Time-Seriea models- it can have heteroscedasticity if the dependent variable changes significantly from the beginning to the end of the series. May be initially compay ABC has very less sell of its one of the product which has got popularity over the time and increased the sell.  
**Pure versus impure heteroscedasticity**  
    * Pure heteroscedasticity- in this type even though model is correct, you observe non-constant variance in the residual plots.
    * impure heteroscedasticity- model is incorrectly specified. Might be an important variable was not considered and left out whose effect in turn appeared in variance of an error term $\epsilon$ .
- while heteroscedasticity doesn't introduce bias in coefficients ($\beta_0,\beta_1$,..), but it makes them less precise. Lower precision increases the likelihood that the coefficient estimates are further from the correct population value.  
- Heteroscedasticity tends to produce p-values that are smaller than they should be. This occurs because heteroscedasticity increases the variance of the coefficient estimates but the OLS procedure does not detect this increase. Consequently, OLS calculates the t-values and F-values using an underestimated amount of variance. This problem can lead you to conclude that a model term is statistically significant when it is actually not significant.   
**Example** (heteroscedasticity.csv  from https://statisticsbyjim.com/regression/heteroscedasticity-regression/)  
if we take population as x and Accidents as Y then, below is the chart prepared in Excel.
![Alt text](residual_excel.png) fig. 5   
we can see huge variations in the scattered plot of the Accidents Vs. residual chart.
- now, can we transform the Accidents to Accident Rate? i.e. (20/147604.989)=0.000135498, this way we can reduce the variance. Let us see how our scatter plot looks now.  population vs Accident Rate  
![Alt text](residual_excel2.png) fig. 6  
##### Incorrect Set of independent Variables:- 
![Alt text](omit1.png)  
See above image, if Y dependent on X and Z is errorneously omitted from regression several result can be noted.  
1. Since Y is regressed on only X, the A+B area is used to estimate $\beta_x$· But the red area reflects variation in Y due to both X and Z, so the resulting estimate of $\beta_x$ will be biased.  
2. if Z has been included in regression, only the 'B' area would have been used in estimating $\beta_x$. Omitting Z thus increases the information used to estimate $\beta_x$ by the area 'A', implying that the resulting estimate, although biased, will have smaller variance. Thus, it is possible that by omitting Z,the mean square error of the estimate of $\beta_x$ may be reduced.   
3.  The magnitude of the Area 'D' reflects the magnitude of $\sigma^2$. But when Z is omitted, $\sigma^2$ is estimated using area 'C+D', resulting overestimate of $\sigma^2$ (i.e., the area 'C' influence of Z is errorneously attributed of the error term). This overestimate of  $\sigma^2$ causes an overestimate of the variance-covariance matrix of the estimate of $\beta_x$.  
4. if Z is orthogonal to X, the are 'A' does not exist, so the bias noted above disappears.

**Weighted Regression**  
- Another way to weigh each observations and tell how  important it should be in the model fit. The idea is to give small weights to observations associated with higher variances to shrink their squared residuals. Weighted regression minimizes the sum of the weighted squared residuals. When we use the correct weights, heteroscedasticity is replaced by homoscedasticity.  
- Weighted regression can be used to:  
  1. Handle non-constant variance of the error terms in linear regression.
  2. Handle variability in measurement accuracy. 
  3. Account for sample misrepresentation and duplicate observations.  

**coefficient of determination**   
- $R^2$ is called the coefficient of determination. This quantity is a measure of the proportion of variability explained by the fitted model.
we know from fig. 4 that   
**SSE=$\sum_{i=1}^{n}(y_i-\hat y)^2$,** The SSE value is the variation
due to error, or variation unexplained. Clearly, if SSE = 0, all variation is
explained. The quantity that represents variation explained is SST - SSE.  
**SST=$\sum_{i=1}^{n}(y_i-\bar y)$**, total corrected sum of squares, the variation in the response values that ideally would be explained by the model.  
 **$R^2= 1- \frac {SSE}{SST}$**, or $\frac {SSR} {SST}$ if all the variations are explained then SSE=0, and $R^2=1$.  
 - There are *Dangers of using $R^2$ for comparing competing models* for same datasets.Adding additional terms to the model (e.g., an additional regressor)decreases SSE and thus increases $R^2$ (or at least does not decrease it). This impliesthat $R^2$ can be made artificially high by an unwise practice of overfitting (i.e., the inclusion of too many model terms). Thus, the inevitable increase in $R^2$ enjoyed by adding an additional term does not imply the additional term was needed. So, it is suffice to say that one should not subscribe to a model selection process that solely involves the consideration of $R^2$.  
  ####  Predictions- Mean Response   
  - **Standard Error**:- it is a standard deviation of the distribution form by sample means.   var= $\frac {\sigma^2}{n}$ --> SE= $\frac {\sigma}{n^{0.5}}$, it shows the variability between the means of the samples.  
  - SE is vitally important in determining whether there is a true relation in Y and X in the population.  

    i.e., when we take a random sample from population we may get different $\hat \beta_i$, it may lead to different regression line. SE decreases as sample size increases, because bigger sample size gives better approximation.  
  - The equation $\hat y = b_0+b_1x $ can be used to predict or estimate the mean response $\mu_{Y|x_0} $ at $x=x_0$.  where $x_0$ is not necessarily one of the prechosen values,or it may be used to predict a single value $y_0$ of the variable $Y_0$, when $x = x_0$. We would expect the error of prediction to be higher in the case of a single predicted value than in the case where a mean is predicted. This, then, will affect the width
of our intervals for the values being predicted. We shall use the point estimator $\hat Y_0 = \beta_0 + \beta_1x_0 $ to estimate $μ_{Y |x_0} =
\beta_0 + \beta_1x$. It can be shown that the sampling distribution of $\hat Y_0$ is normal with mean.  Now variance is   
**$\sigma^2_{Y_0}=\sigma^2_{B_0+B_1x_0}= \sigma^2_{\bar Y + B_1(x_0-\bar x)}=\sigma^2 [ \frac {1}{n}+ \frac {(x_0- \bar x)^2}{S_{xx}}]$**   ........... Eq.1
- and variance for prediction interval is  
 **$\sigma_{\hat Y_0-Y_0}^2 = \sigma^2_{B_0+B_1x0+\epsilon} = \sigma^2_{\bar Y +B_1(x_0-\bar x)-\epsilon_0}= \sigma^2 [1+ \frac {1}{n}+ \frac {(x_0- \bar x)^2}{S_{xx}}] $**  ...........Eq.2 
 - now if we use $x_0=12$, then 95% confidence interval for true mean is   
  **$0.040 \pm 0.182$** and 95% prediction interval for single value of $Y$ is **$0.040 \pm0.396$**  .  
  - if we draw a plot for this we can find that the plot for mean of $Y$ is nearer and narrower to regression line compare to plot of single prediction interval as visible by equation 1 and 2.


- the mean of $Y$ for a given $x$ value, $\mu _{Y|x_0}$ can be calculated using regression equation. For ex. if $\hat Y = -0.3745+0.0345x_0 $ then for $x_0=12$, we have $\hat Y=0.040$. Hence our estimated mean is same as predicted value for $\hat Y$.

- *Derive a slope $\beta_1$ formula when intercept is zero*  
    - we know that we've to minimize the least square estimator by taking derivative and equating with 0 that is   
    $L(\beta_1)= \sum_{i=1}^n (y_i-\beta_1 x_i)^2$  
    $L'(\beta_1)=2 \sum (y_i-\beta_1 x_i)(-x_i)$ , equate with 0,  then   
    $L'(\beta_1)=-2 (\sum(x_iy_i)-\beta_1 \sum x_i^2)=0$  
    $\beta_1 = \frac{\sum(x_iy_i)}{\sum x_i^2} $
    
- one can look at the python code ...  
https://github.com/swetangacharya/DataScience/blob/main/Walpole_ex11_29.py

#### ANOVA  
- if we want to analyze how good our estimated regression line is then we can use the technique of analysis-of-variance. suppose we've $n$ data points in $(x_i,y_i)$ , and for estimating $\sigma^2$ (model error variance, an experimental error variation around regression line) we have identity  
**$S_{yy}=b_1S{xy}+SSE$**  
**$ \sum_{i=1}^n (y_i-\bar y)=\sum_{i=1}^n (\hat y_i-\bar y)+\sum_{i=1}^n (y_i-\hat y)$**  
**$SST=SSR+SSE$**
-  we've partitioned the total corrected sum of squares of $y$ into two components. **SSR**, is called the regression sum of squares,
and it reflects the amount of variation in the y-values explained by the model, i.e., a postulated straight line. **SSE**, sum of square of errors, which reflects variation about the regression line.  
#### explained and unexplained errors .  
- Let us revisit the Fig.4 where we've $(x_i,y_i)$-an actual reading, $\hat Y$- the regression line, and , $\bar Y$- the expected Mean of all $y_i$. now take an Example of selling of icecream for an icecream vendor on a given day.
Say, $x_i$ is a temperature on a given day and $y_i$ is number of icecream sold on that day. Now with regression we are trying to explain why on a given day there is a different value from mean- $\bar Y$. Here we can seperate that in two partition one is explained and another one is unexplained deviation from $\bar Y$. SO, on a given day with given temperature $x_i$ we expected $\hat y_i$ (value that regression line gives), but actual value $y_i$ may differ. that we can't explain?? because our regression line couldn't predicted that high/low $y_i$ (sale of icecream)!  
But we can explain why it $(\hat y_i)$ has deviated from  mean $(\bar y)$, i.e., expected or explained deviation from mean.  
when we sum all variation of unexplained variation from value that is predicted by the regression line equation, i.e.,$(y_i-\hat y)^2=SSE$.
now value explained by regression line is $(\hat y_i-\bar y)^2=SSR$.
Suppose that we are interested in testing the hypothesis...  
$H_0: \beta_1=0 , H_1:\beta_1 \neq0$

##### F value :- 
 The F value is a value on the F distribution. Various statistical tests generate an F value. The value can be used to determine whether the test is statistically significant.

The F value is used in analysis of variance (ANOVA). It is calculated by dividing two mean squares. This calculation determines the ratio of explained variance(SSR)/df1 (i.e.,1 in our case) to unexplained variance (SSE/df2)(i.e., n-1).

The F distribution is a theoretical distribution. There are many of these distributions, and each of them differs based on the degrees of freedom.

The F value and the degrees of freedom of the sources of variance are used to determine the probability of the F value. The probability is the significance value for the test.

#### Lack of fit  
- If a line fits the data well, then the average of the observed responses at each x-value should be close to the predicted response for that x-value. Therefore, to determine how much of the total error is due to lack of model fit, we determine how far the average observed response at each x-value is from the predicted response of each data point. That is we calculate $(\bar y- \hat y_{ij})$, to *quantify total lack of fit*   
**$\sum_i \sum_j (\bar y- \hat y_{ij})^2$, lack of fit sum of squares (SSLF)**  
- To determine how much of the total error is due to just random error, we determine how far each observed response is from the average observed response at each x-value. That is, we calculate the distance $(y_{ij}- \bar y_i)$. To quantify the total pure error, we determine this distance for each data point, square the distance, and add up all of the distances to get:  
**$\sum_i \sum_j (y_{ij}- \bar y_i)^2$, pure error sum of squares (SSPE)**    
 **SSE= SSLF+SSPE**  
 - We break down the residual error ("error sum of squares" — denoted SSE) into two components:
    - a component that is due to lack of model fit ("lack of fit sum of squares" - denoted SSLF)
    - a component that is due to pure random error ("pure error sum of squares" - denoted SSPE)
    - If the lack of fit sum of squares is a large component of the residual error, it suggests that a linear function is inadequate.  
    - important to note here is, *if each x value in the data set is unique, then the lack of fit test can't be conducted on the data set.* Even when we do have replicates, we typically need quite a few for the test to have any power. As such, this test generally only applies to specific types of dataset with plenty of replicates.  
    ![Alt text](lack_of_fit1.png)  fig. 7  
    - we assume that our model is $Y=\beta_0+\beta_1x+\epsilon$, but in reality the model could be $Y==\beta_0+\beta_1x+\beta_2x^2+\epsilon$, as shown in fig.7. we assume that each $x_i$ we have $n_i$ measurement, thus, we've $n_i$ number of measurement at each $y$ for each $x_i$ measurement, in our Example we took $n_i=4, i=1,2,3...,m$, same for all for simplicity. so, $n_1=n_2=n_3=n_m=4$.  
    - the orange regression line can have slightly lower or higher slope, but it may miss the mean of 1 or 2 $y_i$ points.


    #### Bias-Variance tradeoff  
    * **Bias (Underfitting)** - A model with high bias tends to underfit the data, meaning it oversimplifies the underlying patterns. This leads to poor predictive performance as the model cannot capture the complexity of the real-world problem.  A model with high bias will struggle to fit the training data, resulting in low training accuracy. The same model will also perform poorly on the test set, leading to low test accuracy
    * **Variance (Overfitting)** - is the error due to excessive complexity in the learning algorithm. A model with high variance captures not only the underlying patterns but also the noise in the training data. This leads to poor generalization to unseen data.  
      - for Example, a model can fit the training data perfectly, it might perform poorly on new, unseen digits because it has essentially memorized the training examples, including their individual quirks. A model with high variance can fit the training data exceedingly well, achieving high training accuracy. However, this model will perform significantly worse on the test set, exhibiting lower test accuracy
      - Before we dive deeper, we should understand **Interoperability and flexibility**  
      Interoperability and flexibility are interconnected concepts in technology, but they address different aspects of system design and operation.  
      **Interoperability** is the ability of different models or systems to work together seamlessly, ensuring that data can be shared and analyzed consistently across platform.  
      **Flexibility** in regression involves the ability to adapt models to different data types, scenarios, or evolving needs without significant rework. Enhances model accuracy by accommodating complex relationships.Supports rapid adaptation to new data or changing conditions.  
      * Flexible models tend to overfit the training data by capturing not only meaningful patterns but also random noise. This overfitting results in increased variability in predictions across different datasets, thereby increasing variance  
      Suppose we've to fit a model $\hat f(x)$ to some training data $T_r$, and let $(x_0,y_0)$ be a test observation drwwn from the population.
      if the true model **$Y=f(X)+\epsilon$**   (with $f(x)=E(Y|X=x))$.  
    > $$ E(y_0- \hat f(x_0))^2 =Var(\hat f(x_0))+[Bias(\hat f(x_0))]^2+ Var (\epsilon)$$   

    * The expectation averates over the variability of $y_0$ as well as the variability in $T_r$. Note that $Bias(\hat f(x_0))=E[\hat f(x_0)]-f(x_0)$.  
    * Typically as flexibility of $\hat f$ increases, its variance increases, and its bias decreases, So, choosing the flexibility based on average test error amounts to a bias-variance trade-off.


    

   **Nonlinear Regression**
  - When a relationship appears to be nonlinear, it is possible to transform either the dependent variable or one or more of the independent variables so that the substantive relationship remains nonlinear, but the form of the relationship is linear, and can therefore be analyzed using OLS estimation. Another way of saying that a relationship is substantively nonlinear but formally linear is to say that the relationship is nonlinear in terms of its variables but linear in terms of its parameters...    
  ##### Ridge regression (L2 Regression):-    
  it is a linear regression technique that addresses the problem of multicollinearity (high correlation between predictor variables) by adding a penalty term to the cost function. This penalty, also known as L2 regularization, shrinks the coefficients towards zero, but unlike Lasso regression, it doesn't force them to become exactly zero. This makes it suitable for situations where feature selection isn't desired, but reducing the impact of correlated variables is necessary.   
  $ \color{yellow}\hat L(w,x,y)=L(w,x,y)+ \gamma ||w||^2$  
  $\hat L(w;x^{(i)},y^{(i)})= \frac{1}{2N} \sum(w^T x^{(i)}-y^{(i)})^2+ \gamma ||w||^2$

  ##### LASSO (Least Absolute Shrinkage and Selection Operator) regression (L1 Regression):-  
  the regression shrinkage and selection operator tries to combine subset selection and shrinkage (toward zero) estimation by minimizing the sum of squared errors subject to the restriction that the sum of absolute values of the coefficient estimates be less than a specified constant.  
  $ \color{yellow}\hat L(w,x,y)=L(w,x,y)+ \gamma ||w||$





```python showLineNumbers {4} caption="The highlighted line declares a Pandas Series"
import numpy as np
import pandas as pd

x = pd.Series([10, 20, 30, 40, 50], dtype='string')
x.mean()
```

:::info[Want to learn more?]
See our docs on [Code ->](https://www.scipress.io/post/jKgNqkgB1k6saoDUKkiP/Code)
:::