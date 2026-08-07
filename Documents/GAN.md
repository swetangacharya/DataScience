**GAN**  
- GAN has two components, Generator (**G**) and Discriminator(**D**). The Generator creates fake data and tries to convince D that it is real. The Discriminator receives data and tries to determine if it came from real data or was made by G (i.e., a 'Fake' data).
- The generator takes in random data and tries to convert it into something that looks realistic. The dimensions $m$ and $d$ and could be tensors too. For images, $d$ would be something like $d=(C,W,H)$ and $m$ could be $m=(C',W',H')$
-  G works by taking in a latent vector $z \in R^m$,and predicting an output the same size and shape as the real data. We get to choose the number of dimensions $m$ (hyperparameter) for latent variable $z$ : it should be large enough to represent different
concepts in our data (which takes some manual trial and error). For example, some latent properties we might want to learn are smile/frown, hair color, and hairstyle. The values of z are called latent because we never observe what they actually are: the model has to infer them from the data. If we do a good job, the generator G can use this latent representation $z$ as a smaller and more compact representation of the data—kind of like a lossy form of compression.
You can think of the generator as a sketch artist who has to take in a description (the latent vector $z$ ) and, from that, construct an accurate picture for the output! The same way a sketch artist’s drawing is a product of how they interpret your description; the meaning of these latent vectors depends on how the generator G interprets them. In practice,
we use a simple representation of each latent variable being sampled from a Gaussian distribution ( $z_i = N(0,1)$ ).
* **Loss of GAN** :  
	D and G are both computing their loss from D’s output, and G’s loss is somehow the opposite of D’s.   
	$loss_D = l(D(x),y_{real})+ l(D(G(z)),y_{fake})$  
  $loss_G = 0 + l(D(G(z)),y_{real})$  

  $l(D(x),y_{real})$ gives us D’s loss on real data. Its loss on fake data is also a straightforward classification, except we replace the real data $x$ with the generator's output $G(z)$, and we use $y_{fake}$ as the target because the input to **D** is fake data from **G**.  The generator’s output is given to
say that the result looks fake the discriminator, which wants to to compute **G**'s loss.

	On to the generator **G**'s loss. For real data, it is easy: **G** does not care what **D** says about real data, so nothing happens to **G** when real data is used. But for     fake data, **G** does care about what **D** says. It wants **D** to call its fake data real, because that means it has successfully tricked **D** .
	**G** only cares about the fake data, and **G** is successful if it tricks **D** into calling fake data real.  
  $loss_G = E_{z \in N(0,1)} l(D(G(z)), y_{real})$,  where $y_{real}=1 and y_{fake}=0$, so in most of the articles the loss function looks like below.
  $\color{magenta}\overbrace{G}^{min}\overbrace{D}^{max}E _{x \in D(x)} [log(D(x))] + E_{z \in N(0,1)} [log(1-D(G(z)))]$
  <img width="892" height="143" alt="image" src="https://github.com/user-attachments/assets/66002945-38b5-41a4-8911-1147200a5861" />

- Let us recap **KL divergence**
$D_{KL} = \sum _x P(x)log \frac{P(x)}{Q(x)}$,
- Averaging is done according to distribution $P$, it compares how much probability $P(x)$ is assigned to that outcome to the how much probability assigned to same outcome by distribution $Q$. Outcome that is unlikely under $P$ contribute very little.
-  we have log ratio of $P(x)$ and $Q(X)$, so if ratio is large then we can think that the outcomes that are high under $P(x)$ are ignored by (i.e., produced low outcome) $Q(x)$. but it is asymmetric. But this works well when $Q$ assign some probability to every outcome that comes from $P$. because if $Q$ assign zero probability to any outcome then $log \frac{P(x)}{Q(x)}$ becomes $\infty$. keeping this in mind we define new divergence which is called
**Janson-Shannon Divergence**, which is symmetric.
  P={0.2,0.3,0.5} and Q={0.3,0.4,0.3}
then $M =\frac{P+Q}{2}$, that is M ={0.25,0.35,0.4}

$D_{JS}(P||Q) = \frac{D_{KL}(P||M) + D_{KL}(Q||M))}{2}$  
$D_{KL}(P||M)=0.1193, D_{KL}(P||M)=0.1193,D_{KL}(Q||M)=0.1109$  
so, $D_{JS}(P||Q) = 0.1151$,   
which tells in an average there is a 0.1151 bit of difference if we take samples from distribution $Q$ instead of $P$. JSD value is always between [0,1] bits.  

**Wasserstein Distance**   **(Earth mover's distance)**  
The cost of moving dirt is **Cost = mass x distance**  
Wasserstein Distance finds the transport plan that minimizes total cost. key is to move near by mass to near by location
Only transport far when you absolutely must.   
$\color{magenta}W_1(P,Q)= \int_{- \infty}^{ \infty}|F_P(x)-F_Q(x)|dx$


```python
>>> from scipy.stats import wasserstein_distance
>>> x,y=[0,1],[0,2]  # location of observations
>>> p,q=[0.5,0.5],[0.25,0.75] # Probability weights
>>> distance = wasserstein_distance(x,y,p,q)
>>> print(distance)
1.0
```
```python
>>> from scipy.spatial.distance import jensenshannon
>>> import numpy as np
>>> p=np.array([0.2,0.3,0.5])
>>> q=np.array([0.3,0.4,0.3])
>>> distance = jensenshannon(p, q)
>>> print(f"Jensen–Shannon distance: {distance:.4f}")
Jensen–Shannon distance: 0.1458

>>> x,y=[0,1,2],[0,1,2]  # location of observations
>>> wasserstein_dist= wasserstein_distance(x,y,p,q)
>>> print(f"wasserstein_disttance:{wasserstein_dist:.4f} ")
wasserstein_disttance:0.3000
```  
**manual calculation**
$CDF_p= [0.2,0.5,1.0]$  
$CDF_q= [0.3,0.7,1.0]$  
their differences are  
$|02-0.3|=0.1,  |0.5-0.7|=0.2, |1.0-1.0|=0$  
$W_1= 0.1(1-0)+0.2(2-1)=0.3$

  
**Stability of the GAN training.**

- Generative Adversarial Networks, or GANs, are challenging to train.

- The discriminator model must classify a given input image as real (from the dataset) or fake (generated), and the generator model must generate new and plausible images.
The reason GANs are difficult to train is that the architecture involves the simultaneous training of a generator and a discriminator model in a zero-sum game. Stable training requires finding and maintaining an equilibrium between the capabilities of the two models.
The discriminator model is a neural network that learns a binary classification problem, using a sigmoid activation function in the output layer, and is fit using a binary cross entropy loss function. As such, the model predicts a probability that a given input is real (or fake as 1 minus the predicted) as a value between 0 and 1.

- The loss function has the effect of penalizing the model proportionally to how far the predicted probability distribution differs from the expected probability distribution for a given image. This provides the basis for the error that is back propagated through the discriminator and the generator in order to perform better on the next batch.

- The WGAN relaxes the role of the discriminator when training a GAN and proposes the alternative of a critic.
Instead of using a discriminator to classify or predict the probability of generated images as being real or fake, the WGAN changes or replaces the discriminator model with a critic that scores the realness or fakeness of a given image.

- This change is motivated by a mathematical argument that training the generator should seek a minimization of the distance between the distribution of the data observed in the training dataset and the distribution observed in generated examples. The argument contrasts different distribution distance measures, such as Kullback-Leibler (KL) divergence, Jensen-Shannon (JS) divergence, and the Earth-Mover (EM) distance, referred to as Wasserstein distance.
  


> Ref: Edward Raff  
