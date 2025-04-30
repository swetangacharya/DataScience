import numpy as np
import math
from scipy.stats import entropy
from scipy.special import rel_entr,kl_div


def KL(a,b):
    a=np.asarray(a)
    b=np.asarray(b)
    return np.sum(np.where(a!=0, a*np.log2(a/b),0))



# Example joint probability table for X and Y
# columns: X=0, X=1; rows: Y=0, Y=1,Y=2

joint_prob = np.array([[0.1, 0.2],[0.4, 0.1],[0.1,0.1]])
#joint_prob = np.array([[0.1, 0.2],[0.3, 0.0],[0.0,0.4]])


# Marginal distribution for Y: sum over columns
p_Y = np.sum(joint_prob, axis=1)
# Marginal distribution for X: sum over rows
p_X = np.sum(joint_prob, axis=0)
print('Marginal probability when x=0 and x=1', p_X)
print('Marginal probability when y=0,y=1,y=2', p_Y)

# Calculate individual entropies
H_X = entropy(p_X, base=2)
H_Y = entropy(p_Y, base=2)

print("Entropy of X  H(X):", H_X)
print("Entropy of Y  H(Y):", H_Y)
print('-'*35)
cond_entropy1=0.0
cond_entropy2=0.0
x_0=p_X[0]
x_1=p_X[1]
p_y_x0=[]
p_y_x1=[]

Hy_x0,Hy_x1=[],[]
for i in joint_prob[:,0]:
    
    if i>0:
        p_y_given_x0=i/x_0
        p_y_x0.append(p_y_given_x0)
        f1=p_y_given_x0 * math.log2(p_y_given_x0)
        Hy_x0.append(-f1)
        cond_entropy1+=f1
    elif i==0:
        p_y_x0.append(0.0000000001)


for j in joint_prob[:,1]:
    if j>0:
        p_y_given_x1=j/x_1
        p_y_x1.append(p_y_given_x1)
        f2=p_y_given_x1 * math.log2(p_y_given_x1)
        Hy_x1.append(-1*f2)
        cond_entropy2+=f2
    elif j==0:
        p_y_x1.append(0.0000000001)
        
     
print('conditional entropy1 H(Y|x=0) is ', -cond_entropy1)
print('conditional entropy2 H(Y|x=1) is ', -cond_entropy2)
print(   '---'*25)
print('Total conditional entropy H(Y|X)= marginal_probability when(x=0) * conditional_entropy(H(Y|x=0)) + marginal_probability when (x=1) * conditional_entropy(H(Y|x=1))')
print('Total Conditional entropy = P(x=0)*H(Y|x=0)+P(x=1)*H(Y|x=1) = ', "{:.2f}".format(-(p_X[1]*cond_entropy2+p_X[0]*cond_entropy1)), ' that is less than H(Y) ' , "{:.2f}".format(H_Y))
print(   '---'*25)
print('Conditional probability y|x=0 and x|x=1 is ', p_y_x0,p_y_x1)

P1=joint_prob[:,0]
#P1=np.array([0.1,0.4,0.1])
P1=P1/0.6
P2=joint_prob[:,1]
#P2=np.array([0.2,0.1,0.1])
P2=P2/0.4

print(   '---'*25)
#print('P1 and P2',P1,P2)
print('KL_Div(Y|x=0 || Y|x==1) is = ',  KL(P1,P2))
print('Another way to calculate KL_div is by entropy funtion in scipy.stats.entropy')
print(entropy(P1,P2,base=2))

