import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
np.set_printoptions(precision=3)

# Example data 
x = np.array([2, 15, 30, 10, 20])
X=np.vstack(x)
y = np.array([7,50,100,40,70])

df = pd.DataFrame({'x': x, 'y': y})

# Plot the data
"""
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sample Data for Interval Demonstration')
plt.show()
"""
# fit a LR model
model=LinearRegression(fit_intercept=False)  # Y intercept=0
model.fit(X,y)
y_pred=model.predict(X)

# draw scatter plot 
plt.figure(figsize=(8, 6))
sns.regplot(x='x', y='y', data=df)
plt.text(min(x), max(y), 'y = {:.2f} + {:.2f}x'.format(model.intercept_, model.coef_[0]), ha='left', va='top')
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Regression Plot")
plt.show()

# calculate MSE and standard error
mse=mean_squared_error(y,y_pred)
std_error=np.sqrt(mse)

print('coefficiend of model ', model.coef_)
print('Y intercept is ', model.intercept_)
print('predicted Y',y_pred)
print('original Y is ', y)
print('mean square error is', mse)
print('standard error is ', std_error)
