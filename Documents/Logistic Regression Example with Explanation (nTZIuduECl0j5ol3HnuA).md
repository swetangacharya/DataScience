---
title: Logistic Regression Example with Explanation
description: Template post about dogs; They're like humans, but better.
slug: Logistic-Regression
is_draft: true
icon: shopping-cart
tags:
  - confidence interval
  - wald test
  - deviance
  - significance of coefficients
---
- Example Data at github: https://github.com/swetangacharya/DataScience  
 file: heartDisease.csv

```python showLineNumbers {7} caption="The highlighted line removes index" 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


file_path='heartDisease.csv'
df1=pd.read_csv(file_path,sep=' ')
df=df1.drop(columns=df1.columns[0])  

# Group by agegrp and aggregate
summary = df.groupby('agegrp').agg(
    #rows_count = ('id', 'count'),
    rows_count=('age','size'),
    #age_range=('age',['min','max']), # doesn't work
    age_min=('age','min'),
    age_max=('age','max'),
    
    chd_1_count = ('chd', lambda x: (x == 1).sum())
)
## combine min max age
summary['age_range'] = summary['age_min'].astype(str) + '-' + summary['age_max'].astype(str)
summary = summary.drop(columns=['age_min', 'age_max'])


# Calculate the mean
summary['mean'] = summary['chd_1_count'] / summary['rows_count']

# Reset index for a clean table
summary = summary.reset_index()

# print summary table
print(summary, '\n', summary['mean'].sum()/summary['rows_count'].sum())

# using statsmodels 
X=df.iloc[:,0:1]
y=df['chd']  
X1=sm.add_constant(X)
model=sm.Logit(y,X1)
res=model.fit()

print(res.summary())
deviance= 2*(res.llf-res.llnull)
print('deviance=', deviance)
print('p_values=', res.pvalues)
print('t-values(z-values)', res.tvalues)
print('Confidence intervals',res.conf_int())
print('standard errors', res.bse)
print('Coefficients',res.params)



## output of the summary table print ###
   agegrp  rows_count  chd_1_count age_range      mean
0       1          10            1     20-29  0.100000
1       2          15            2     30-34  0.133333
2       3          12            3     35-39  0.250000
3       4          15            5     40-44  0.333333
4       5          13            6     45-49  0.461538
5       6           8            5     50-54  0.625000
6       7          17           13     55-59  0.764706
7       8          10            8     60-69  0.800000 

## output of print(res.summary())
==============================================================================
Dep. Variable:                    chd   No. Observations:                  100
Model:                          Logit   Df Residuals:                       98
Method:                           MLE   Df Model:                            1
Date:                Sat, 05 Jul 2025   Pseudo R-squ.:                  0.2145
Time:                        10:38:09   Log-Likelihood:                -53.677
converged:                       True   LL-Null:                       -68.331
Covariance Type:            nonrobust   LLR p-value:                 6.168e-08
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -5.3095      1.134     -4.683      0.000      -7.531      -3.088
age            0.1109      0.024      4.610      0.000       0.064       0.158
=============================================================================
deviance= 29.309890034459826

p_values= const    0.000003
age      0.000004
dtype: float64

t-values(z-values) const   -4.683484
age      4.610220
dtype: float64

Confidence intervals               0         1
const -7.531376 -3.087531
age    0.063765  0.158078

standard errors const    1.133655
age      0.024060
dtype: float64

Coefficients const   -5.309453
age      0.110921
dtype: float64

```
- Log-Likelihood: (-53.677), It shows- How likely the observed set of outcomes(both '1' and '0') is given set of input values and a specific model.

- **deviance** = 29.309, it measures the deviance of the fitted logistic model with respect to a perfect model. $D=-2log-Likelihood(\hat \beta_1)$, here $(\beta_1=age)$  
![Alt text](saturated_model.png)  
- The null deviance serves for comparing how much the model has improved by adding the predictors.   
$\color{blue}R^2= 1- \frac {D}{D_0}=1 - \frac{deviance(fitted logistic,saturated model)}{deviance(null model, saturated model)}$  
- **Saturated model:-** In logistic regression, a saturated model is a model that has as many parameters as there are data points, allowing it to fit the data perfectly. This means the model can capture all the variance in the data, with each observation getting its own parameter or fitted value. As a result, the predicted probabilities match the observed outcomes exactly for every data point.   
- **No residual deviance**: Since the fit is perfect, the deviance (a measure of model fit) is zero for the saturated model. So if D=0, then $R^2=1$, and If the predictors do not add anything to the regression,
then $D=D_0$ and $R^2=0$.  
- in out case(look at the table above) $R^2= 1- \frac{Log-Likelihood}{LL-Null}= 1 - \frac{-53.677}{-68.331}=0.2145$ , which indicates that our variable 'age' is reasonably significant.
:::warning[A warning about $R^2$ in logistic regression]
- It is not the percentage of variance explained by the logistic model, but rather a ratio indicating how close is the fit to being perfect or the worst.
- It is not related to any correlation coefficient.  
:::

$\color{blue}P[\chi^2(1)> 29.31] <\ 0.001$,  
which means age is an significant Variable and it contributes predicting CHD.

In this example, the log-likelihood for the model containing only a constant term is -68.331. 
Fitting a model containing the independent variable (age) along with constant term results in log-likelihood of -53.677.
multiply the difference of them by -2 gives 29.31, i.e.,(deviance)$-2 \cdot (-53.677- (-68.331))=29.31$  
- There are two other statistically equivalent tests: the Wald test and the Score test.  
- The Wald test is equal to the ratio of the maximum likelihood estimate of the slope parameter, $\hat \beta_1$, to an estimate of its standard error.  
$W=\frac{\hat \beta_1}{\widehat{SE}}=\frac{0.111}{0.024}=4.61$

##### confidence interval for slope parameter. which is given in the table  
[0.064,0.158], how it is calculated?
$\hat \beta_1 \pm z_{1-\alpha/2} \widehat SE(\hat \beta_1)=0.111 \pm 1.96 * 0.0241= [0.064,0.158]$,  
- the results suggest that the change in the log-odds of CHD per one year increase
in age is 0.111 and the change could be as little as 0.064 or as much as 0.158 with
95 percent confidence.


----------------------------------------------------------

#### Multinomial logistic regression:-   

- suppose there are 2 covariates, $X_1$ and $X_2$,
$log(odds(Y=1|X))=\beta_0+\beta_1 x_1+\beta_2 x_2$,  
$\beta_1$ is the difference between log odds(Y=1) for two populations that differ in X1 by one unit and have the same $X_2$, in other words  $\beta_1$ is the change in the log odds(Y=1) when $X_1$ increases one unit and $x_2$, remains fixed.
- Child1 $(Y_1,x_1,x_2)$ and Child2 $(Y_2,(x_1+1),x_2)$  
log odds $(Y_2=1)= \beta_0+\beta_1(x_1+1)+\beta_2x_2$  
log odds $(Y_1=1)= \beta_0+\beta_1 x_1+\beta_2x_2$ , difference in log odds is $\beta_1$, as explained above.


------------------------------------------------------------
:::info
**New here?**  
Every post begins with a random template to help you start writing.

**Still confused?** 🤔  
- [:icon[video] Watch the **Getting Started** video :icon[external-link]](https://www.youtube.com/watch?v=0h4gRvgoRn4&list=PL9Zhnnyw1lVND99JOWpTyYly9heBP4-Xh){target=_blank}
- [:icon[file-text] Read the **Getting Started** guide](https://www.scipress.io/post/l7R0XuDTe6R1dC2dS5cc/Getting-Started)
:::

---

![Golden Retriever]{ .px-12 .w-96 .mx-auto }

*Dogs*, our loyal companions, embody _unconditional love_ and _empathy_. Their unwavering loyalty and understanding create deep bonds with humans. 

## Reasons why dogs are amazing

- Unwavering loyalty and companionship
- Unconditional love and empathy
- Versatility as working partners

**Dogs truly deserve the title "man's best friend."**

<!-- Image references -->
[Golden Retriever]: https://images.unsplash.com/photo-1612774412771-005ed8e861d2?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80