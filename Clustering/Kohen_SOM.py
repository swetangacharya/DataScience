import numpy as np
import math
class SOM:
    def winner(self, weights, sample):
        D0 = 0
        D1 = 0

        w1=np.array(weights[0])
        t1=np.array(sample)
        w2=np.array(weights[1])
        D0=sum(np.square(w1-t1))
        D1=sum(np.square(w2-t1))
        '''
        for i in range(len(sample)):
            D0 += math.pow((sample[i] - weights[0][i]), 2)
            D1 += math.pow((sample[i] - weights[1][i]), 2)
        print(f' iteration = {i}, and D0={D0},D1={D1}')
        '''
        #print(f' D0={D0},D1={D1}')
        return 0 if D0 < D1 else 1
        
    def update(self, weights, sample, J, alpha):
        ## info ## W(new)=W(old)+ alpha(X-w(old))
        for i in range(len(weights[0])):
            weights[J][i] = weights[J][i] + alpha * (sample[i] - weights[J][i])
        #print('weights',weights)
        return weights
def main():

    ## info ## p-dimensional(4 dimentional observation, also scaled to be in (0, 1), is presented with the values of this weight vector.
    T = [[1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1]]
    m, n = len(T), len(T[0])

    ##info## p-dimensional (4 dimentional weight vectors associated with neurons, each of the values of which is initially random and in the interval (0, 1).
    weights = [[0.3, 0.4, 0.6, 0.8], [0.7, 0.3, 0.6, 0.3]]

    ob = SOM()
    epochs = 3
    alpha = 0.6
    # Inside the "main" function
    for i in range(epochs):
        for j in range(m):
            sample = T[j]
            
            ## info ## The Euclidean distance (or some other preferred distance measure) is calculated between the observation and the vector associated with each neuron
            J = ob.winner(weights, sample)
            #print('J is', J)

            ## info ## The neuron with the smallest distance (the ‘winner’) is then updated, as are a small neighbourhood of neurons around the ‘winner’. The winner’s weight vector wold is brought closer to the input patterns x as follows:
            weights = ob.update(weights, sample, J, alpha)
    # Inside the "Main" function
    s = [1, 0, 0, 0]
    print('-'*30)
    J = ob.winner(weights, s)

    print("Test Sample s belongs to Cluster: ", J)
    print("Trained weights: ", weights)

if __name__ == "__main__":
    main()
