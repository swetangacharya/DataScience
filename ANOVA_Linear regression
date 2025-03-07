import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

"""
    we are trying to find if there is any dependency between
    x and y variable.
    H_0= there is no dependency, Y=B0
    H_1= there is a dependency, Y=B0+B1x
    Below data is taken from Ronald Walpole statistics book, ex. 11.5
--------output looks like --------
SSE: 3.6017272727272784
SSR: 3.6000909090909086
SST: 7.201818181818183
df is  9
F value is , 8.995911052777693
 we can reject the null hypothesis as there are evidence that x and y are dependent
------------------------------------------------
"""
x=np.array([1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2])
X=np.vstack(x)
Y=np.array([8.1,7.8,8.5,9.8,9.5,8.9,8.6,10.2,9.3,9.2,10.5])

# Create and fit the model
reg = LinearRegression()
reg.fit(X, Y)

y_pred=reg.predict(X)

# Calculate SSE
sse = np.sum(np.square(Y - y_pred))
print(f"SSE: {sse}")

# Calculate SSR
y_mean = np.mean(Y)
ssr = np.sum(np.square(y_pred - y_mean))
print(f"SSR: {ssr}")

# Calculate SST
sst = np.sum(np.square(Y - y_mean))
print(f"SST: {sst}")
assert np.isclose(sst, ssr + sse), "SST does not equal SSR + SSE"

#F value
df=len(Y)-2
print('df is ',df)
F=(ssr/1)/(sse/df)
print(f"F value is ', {F}")
F_crit=5.12 # alpha=0.05 and df1=1,df2=9
if F>F_crit:
    print(' we can reject the null hypothesis')
else:
    print("can't reject the null hypothesis as there are not enough evidence of relation of x and y")
