import numpy as np
import math
from scipy.stats import entropy
from scipy.special import rel_entr,kl_div

A=np.array([[0.094,0.117,0.016,0.0],[0.031,0.164,0.078,0.0],[0.0,0.078,0.164,0.031],[0.0,0.016,0.117,0.094]])
#print(A, len(A), len(A[0]))

p_Y=np.sum(A,axis=1)
p_X=np.sum(A,axis=0)

print('Marginal probability p_X= ', p_X)
print('Marginal probability p_Y= ', p_Y)
print('---'*30)
sum1=0
for r in range(len(A)):
    for c in range(len(A[0])):
        if A[r][c]>0:
          sum1+= -A[r][c]*math.log2(A[r][c])

print('---'*30)
print('Joint Entropy H(X,Y) is equal to ', sum1)
                              
                   
Entropy_Y= entropy(p_Y,base=2)
Entropy_X= entropy(p_X,base=2)

print('---'*30)
print("Entropy for X is ", Entropy_X)
print("Entropy for Y is ", Entropy_Y)

print('---'*30)
print("A joint entropy value of", "{:.3f}".format(sum1), " means that, on average, only ", "{:.3f}".format(sum1) ," bits are needed to encode the joint outcome, reflecting very little uncertainty about the pair of variables")

mutual_information= (Entropy_Y+Entropy_X-sum1)

print('---'*30)
print('mutual information is ', mutual_information)
print('A mutual information value of 0.5 bits means that knowing the value of one variable (say,X) reduces the uncertainty about the other variable (Y) by 0.5 bits, on average. In practical terms, this indicates a moderate association between the two variables: they are neither independent (which would yield mutual information of 0) nor perfectly dependent (which would yield mutual information equal to the entropy of the variable with less uncertainty)')
print('---'*30)
print("Some information is shared, but not enough to reliably predict one variable from the other. The reduction in uncertainty is partial, and you can distinguish about 2^0.5 = 1.41, effective levels of the other variable based on knowledge of one.")
print('---'*30)
print("When considered in terms of output entropy, this implies that only 0.256 (i.e. I(X,Y)/H(Y)=0.509/1.99 ) of the output entropy is information about the input and the remainder is just channel noise.")
