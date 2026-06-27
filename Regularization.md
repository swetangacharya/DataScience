In Least square solution, the model tries to minimize the sum of squared errors between the predicted and actual values. However, when the model is too complex or when there are many correlated features, it can fit the training data too closely, capturing noise rather than the underlying pattern.
>**Regularization** addresses this by adding a penalty for large coefficient values, effectively constraining the model. This encourages simpler models that generalize better to new data. Common types include LASSO (L1) and Ridge (L1), which differ in how they penalize coefficients.  

LASSO (Least Absolute Shrinkage and Selection Operator, also called L1 regularization because it adds a penalty proportional to the L1 norm of the coefficients (the sum of their absolute values))
In simple terms, LASSO helps a regression model avoid overfitting by keeping coefficients small and can drive some to zero, which means the model ignores those features. This helps the model focus on the most useful information and makes it simpler and easier to interpret.

One of its main strengths is that it can automatically perform feature selection by shrinking some coefficients to zero. This means the model becomes simpler and more interpretable, as it focuses on the most important features. L1 regularization is also effective at preventing overfitting, especially when dealing with high-dimensional data where the number of features may be large compared to the number of observations. By encouraging sparsity, LASSO helps create models that typically generalize better to new data. The intercept $\beta_0$ is not penalized.
However, L1 regularization also has some drawbacks. When features are highly correlated, LASSO may arbitrarily select one feature and ignore the others, which can make the model unstable and less reliable for interpretation. Additionally, if all features are actually important for predicting the target, L1 regularization might shrink some useful coefficients to zero, potentially reducing the model's predictive performance.  
Unlike ordinary least squares or ridge regression, LASSO does not have a simple closed-form solution and requires iterative optimization, which can be computationally more intensive. The effectiveness of L1 regularization depends on the choice of the regularization parameter ($\lambda$), which should be carefully tuned to achieve good results.  
**Formula**  
The goal is to find the set of coefficients that fit the data well (by minimizing the sum of squared errors) while keeping the model simple by penalizing large coefficients. In other words, LASSO seeks to minimize a loss function that combines the usual error term with an additional penalty based on the absolute values of the coefficients.  
 $$SSE + \lambda \sum_{j=1}^p |\beta_j| $$  
  1. $\lambda $: regularization parameter(control the strength of penalty)
  2. p: number of features in the model
  3. $\beta_j$: coefficient for the $j^{th}$ coefficient, ensure non negative penalty)
  4. **SSE**: sum of square errors between predicted and actual value.

$$ \left\{SSE+ \lambda \sum_{j=1}^p |\beta_j| \right\}$$

$\left\ {SSE+ \lambda \sum_{j=1}^p |\beta_j| \right\ }$


  
