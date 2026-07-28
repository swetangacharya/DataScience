---
title: Artificial Neural Networks (Template)
description: Template post about code blocks
slug: code-blocks
is_draft: true
tags:
  - ANN
  - SOM (self organizing maps)
  - Kohonen
---

:::info
**What is ANN?**  
The three essential features of a neural network are the basic computing elements usually referred to as neurons, the network architecture describing the connections between computing units, and the training algorithm used to find values of the network parameters for performing a particular task.  

Regarding supervised learning, the most common architectures are considered, such as Multilayer Perceptron,
Radial Basis Network, ADALINE Networks, HOPFIELD Networks, Probabilistic Networks, Linear Networks, Generalised Regression
Networks, LVQ Networks, Linear Networks and Networks for Regression Model Optimisation.

Unsupervised learning develops Pattern Recognition and Cluster Analysis Networks such as KOHONEN Networks (SOM Self-Organising
Maps), Pattern Recognition Networks, Autoencoder Neural Networks, Transfer Learning Networks, Anomaly Detection Networks and
Convolutional Neural Networks.  

Introduction to Neural Networks:- Neural network theory is inspired by the natural neural network of the
human nervous system. A neural network can be defined as a computer system consisting of a number of simple, highly interconnected processing
elements, which process information by their dynamic state response to external inputs.

Adaptive learning: The ability to learn to perform tasks based on input from training or initial experience.
Self-organisation: An ANN can create its own organisation or representation of the information it receives during learning time
Real-time operation: ANN computations can be performed in parallel, and special hardware devices are being designed and manufactured to tak advantage of this capability

Neural networks process information in a similar way to the human
brain. The network is composed of a large number of highly interconnected processing elements (neurons) that work in parallel to solve a specific problem. Neural networks learn by example. They cannot be programmed to perform a specific task. Examples must be carefully selected, otherwise useful time is lost or, worse, the network may operate
incorrectly. The disadvantage is that, because the network discovers by
itself how to solve the problem, its performance can be unpredictable.

**Still confused?** 🤔  
- [:icon[video] Watch the **Getting Started** video :icon[external-link]](https://www.youtube.com/watch?v=0h4gRvgoRn4&list=PL9Zhnnyw1lVND99JOWpTyYly9heBP4-Xh){target=_blank}
- [:icon[file-text] Read the **Getting Started** guide](https://www.scipress.io/post/l7R0XuDTe6R1dC2dS5cc/Getting-Started)
:::

---

##### Introduction to Neural Networks:-  (Techniques_and_Tools_for_AI)
Neural network theory is inspired by the natural neural network of the human nervous system. A neural network can be defined as a computer system consisting of a number of simple, highly interconnected processing
elements, which process information by their dynamic state response to external inputs.

Every neural network possesses knowledge that is contained in the values of the connection weights. The modification of the knowledge stored in the network according to experience implies a learning rule to change the values of the weights
The information is stored in the weight matrix W of a neural network. Learning consists of determining the weights. Depending on the way in
which learning takes place, we can distinguish two broad categories of neural networks: 
Fixed networks in which the weights cannot be modified, i.e. $ \frac{dW}{dt}=0$. In these networks, the weights are fixed a priori according to the problem
to be solved.
Adaptive networks that are able to change their weights, i.e. \frac{dW}{dt} \ne 0$.
**All learning methods** used for adaptive neural networks can be
classified into two broad categories: supervised and unsupervised.
- **Supervised learning** that incorporates an external teacher, so that each
output unit is told what its desired response to input signals should be.
Global feedback may be required during the learning process. Supervised
learning paradigms include **error correction learning, reinforcement
learning and stochastic learning.**    
  An important issue in supervised learning is the problem of error
convergence, i.e. the minimisation of the error between the desired and
calculated unit values. The goal is to determine a set of weights that
minimises the error. A well-known method, which is common to many
learning paradigms, is least mean square (LMS) convergence.  
- **Unsupervised learning** does not use any external master and relies
solely on local information. It is also called **self-organising,** in the sense
that it self-organises the data presented to the network and detects its
collective emergent properties. Unsupervised learning paradigms are
**Hebbian learning and competitive learning.**

##### Self organizing Maps: (SOM)(Kohonen Maps):-
- Self-Organizing Maps (SOMs), also known as Kohonen maps, are a type of
unsupervised learning algorithm that provides a way to visualize high dimensional data in lower-dimensional spaces, typically two dimensions.
SOMs operate on the principles of competitive learning, where neurons in a
network compete with each other to become activated (or "fired") by input
data.   
-  
- The network contains two layers:
    -  an input layer consisting of p-dimensional observations x;
    - an output layer (represented by a grid) consisting of k nodes for the k clusters, each of which is associated with a p-dimensional weight w 
- **Neurons:-** The fundamental units of a Kohonen Map, each neuron represents a point in the input data space. During training, neurons adjust their weights to become more similar to the input data they represent.  
- **Weights:-** Each neuron has an associated weight vector that defines its position in the input data space. These weights are updated during training to reflect the data it’s exposed to, helping the map organize and classify data points.  
- **Grid Structure:-** The layout in which neurons are arranged, typically in a 2D or 3D grid. This structure enables the Kohonen Map to preserve the topological relationships between the input data, making it easier to visualize clusters and patterns.
- When an input is presented, the neuron closest to it (**the Best Matching Unit, or BMU**) is selected. This BMU, along with its neighboring neurons, then adjusts their weights to better match the input, which is where the neighborhood function comes into play.

- The neighborhood function defines how much neighboring neurons are influenced by the BMU, gradually refining the map and preserving the topological relationships in the data.  
- As the neurons adjust and organize themselves during training, the neighborhood function plays a crucial role in shaping the map’s structure. This adjustment process is influenced by the neighbor topologies, which define how neurons are connected and interact with each other.  
- for more detail use below link: https://www.upgrad.com/blog/what-is-kohonen-map/
![Alt text](kohonen_SOM1.png)  
- Classification (clustering) occurs where an input vector is assigned to an output node. Operationally, each output node has a p-dimensional vector of synaptic weights w. The output node is initially assigned a random weight; as the network learns, the input cluster points are provisionally assigned to clusters and the weights are modified. The iterative process eventually stabilizes with the weights corresponding to cluster centres in such a way that clusters that are similar to one another are situated close together on the map.  
- The SOM method thus makes the surface of the neurons recreate (i.e. change
associated weight values) in accordance with the outside world as represented
by the input vectors. In more mathematical terms the process can be described as follows.  
- Consider p-dimensional weight vectors associated with neurons, each of the
values of which is initially random and in the interval (0, 1).
- A p-dimensional observation, also scaled to be in (0, 1), is presented with the values of this weight vector.
- The Euclidean distance (or some other preferred distance measure) is calculated between the observation and the vector associated with each neuron.  
- The neuron with the smallest distance (the"winner") is then updated, as are a small neighbourhood of neurons around the "winner". The winner's weight
vector wold is brought closer to the input patterns x as follows:


###### Here's a code block for sample Kohonen Network.


```python 
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

```

:::info[Want to learn more?]
See our docs on [Code ->](https://www.scipress.io/post/jKgNqkgB1k6saoDUKkiP/Code)
:::