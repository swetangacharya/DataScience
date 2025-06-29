import numpy as np
from scipy.sparse import diags,csr_matrix

def fisher_information_matrix(X, theta):
    """
    Computes the Fisher Information Matrix for logistic regression.
    
    Parameters:
        X     : (n, d) data matrix
        theta : (d,) parameter vector
        
    Returns:
        FIM   : (d, d) Fisher Information Matrix
    """
    # Predicted probabilities
    logits = X @ theta
    y_pred = 1 / (1 + np.exp(-logits))
    
    # Diagonal matrix of variances
    V = diags(y_pred * (1 - y_pred))
    
    # Fisher Information Matrix
    FIM = X.T @ V @ X
    sparse_matrix=csr_matrix(FIM)
    dense_array=sparse_matrix.toarray()
    #return FIM.toarray()  # Convert to dense array if needed
    return dense_array
# Example usage:
np.random.seed(42)
n, d = 100, 3
X = np.random.randn(n, d)
theta = np.random.randn(d)
FIM = fisher_information_matrix(X, theta)
print(FIM)

#----output is 3x3 matrix---https://www.scipress.io/editpost/eBxbEUiDeeiFIKKt8lZE (get the more information here)
#[[10.03908079 -3.15867196  1.43274756]
# [-3.15867196 18.39063302 -0.29376067]
# [ 1.43274756 -0.29376067 18.20723775]]

print(np.linalg.inv(FIM))
## below is the CRLB for each of 3 variables..
#array([[ 0.10647819,  0.01815894, -0.0080859 ],
#       [ 0.01815894,  0.05748637, -0.00050145],
#       [-0.0080859 , -0.00050145,  0.05555141]])
