---
title: Bayesian Statstics
description: Template post about dogs; They're like humans, but better.
slug: why-dogs-are-amazing
is_draft: true
icon: dog
tags:
  - Bayes Theorem
  - Maximum likelihood
  
---

:::info
**Why Bayesian?**  
Frequentist always go under the banner: ' Let the Data Speaks for themselves!'  



**Still confused?** 🤔  
- [:icon[video] Watch the **Getting Started** video :icon[external-link]](https://www.youtube.com/watch?v=0h4gRvgoRn4&list=PL9Zhnnyw1lVND99JOWpTyYly9heBP4-Xh){target=_blank}
- [:icon[file-text] Read the **Getting Started** guide](https://www.scipress.io/post/l7R0XuDTe6R1dC2dS5cc/Getting-Started)
:::

---
##### Probability basics:  
- Suppose we have a biased coin and it is represented by R.V,  $\Theta$, which can adopt two values $\theta_{0.1}$ and $\theta_{0.9}$  
$\color{brown}p(\Theta) = \{p(\theta_{0.1}),p(\theta_{0.9})\} = \{0.75,0.25\}$,  
- if we had a container in which 25% coins have a bias of 0.9, and 75% 
have a bias of 0.1, so that $p(\theta_{0.9})$, is the proportion of coins with a bias of 0.9, and $p(\theta_{0.1})$ is the proportion of coins with a bias of 0.1.  
- If the bias of a coin is $\theta$ then the probability of a head $X_h$ is, by definition, $p(x_h|\theta)=\theta$, and given that th eprobability of a head is the probability that $X=x_h$, the quantity $p(x_h|\theta)$, when written in full is,
$p(X=x_h|\Theta=\theta)=\theta$ , and interpreted as “The probability that the random variable $X =x_h$ (i.e., the probability that the coin lands heads up) given that the random variable $\Theta$ has the value $\theta$, is equal to $\theta$.” This may appear more transparent if we set the coin bias to a specific value, like $\theta = 0.9$, then  
$p(X=x_h|\Theta=0.9)=0.9$, which is interpreted as: “The probability that a coin lands heads up given that it has a bias of 0.9 is 0.9."  
 **- Joint Probability:-**  
 As stated above, in our container of coins, some have 
a bias $\theta_{0.9} = 0.9$, and some have a bias $\theta_{0.1} = 0.1$. As there are only two possible values for the coin bias, the total probability $p(x_h)$ that a  coin flip outcome yields a head is the sum of two joint probabilities. 
  - **Sum Rule:-**
    - $P(x_h, \theta_{0.9})$, the joint probability of observing a head and that a randomly chosen coin has a bias of 0.9, plus  
    - $P(x_h, \theta_{0.1})$the joint probability of observing a head and that  randomly chosen coin has a bias of 0.1.  
    - In other words, the probability that a coin flip outcome is a head is  
      $\color{brown}p(x_h)= P(x_h, \theta_{0.9})+ P(x_h, \theta_{0.1})$  
  
  - **Product Rule:-**  
    - For now, Let us think it as [the probability that a coin has a bias of $\theta_{0.9}$ and that a head is observed] equals [the probability that a coin has a bias of $\theta_{0.9}$ given that it lands heads up] multiplied by \[the probability of observing a head\]  
    $\color{purple}p(x_h, \theta_{0.9})= p(\theta_{0.9}|x_h)p(x_h)$  

Let's take coins N=100, of which 25 has bias of $\theta_{0.9}$ for head, and 75 has  $\theta_{0.1}$. we choose 1 coin, toss it, note if we get head or tail and put it back.  
**Joint probability:-** where we pick the  $\theta_{0.9}$ coin and after flipping we get head is $p(x_h, \theta_{0.9})$.  
This procedure would normally ensure that the bias of the first coin chosen has no effect on the flip outcome of the next coin chosen ; so that bias and head/tail outcome within each pair would be independent. However, to make matters more interesting (albeit a little contrived),

we will assume that there is a slight tendency for the same coin to be 
chosen on consecutive draws. This effectively introduces a correlation 
between the bias of one coin and the flip outcome of the next coin chosen, as shown in Table below. (Despite this correlation, the overall 
proportion of $(\theta_{0.9})$ coin biases chosen will still be equal to 0.25).     
![Alt text](joint_probability1.png)
From the table we can see that probability of getting head irrespective of coin is $p(x_h)=p(x_h,\theta_{0.9})+p(x_h,\theta_{0.1})=0.300$  

**posterior probability:** imagine we don't know the bias of coin and we want to estimate the probability that its bias is 0.9(i.e., from result of flips we want to determine which coin would have selected, so we want to determine $p(\theta_{0.9}|x_h)$, i.e., what is the probability given we get head and it is come from biased coin $\theta_{0.9}$?)
$p(\theta_{0.9}|x_h)= p(x_h, \theta_{0.9})/p(x_h)= 0.225/0.3=0.75$  

**Likelihood:-** How likely it is to observe head given we selected a biased coin $\theta_{0.9}$. i.e., $p(x_h|\theta_{0.9})= p(x_h, \theta_{0.9})/p(\theta_{0.9})= 0.225/0.250=0.9$

$\color{blue}\frac{p(\theta_{0.9}|x_h)}{p(\theta_{0.1}|x_h)}= \frac{0.75}{0.25}=\frac{3}{1}$, this tells if a single coin flip yields a head then it is three times more probable that $\Theta = 0.9$ than it is that $\Theta = 0.1$.   
![Alt text](bayes_venn.png) 
 
 From above Venn Diagram, the probability that the coin has a bias of $\theta_{0.9}$ given that a head $x_h$ is observed is the area $b$ expressed as a proportion of $a$.   
$\color{blue}p(\theta_{0.9}|x_h) = b/a$  
$\color{blue}p(\theta_{0.9}|x_h)p(x_h) = p(x_h, \theta_{0.9})=p(x_h|\theta_{0.9})p(\theta_{0.9})$,  now divide both side by $x_h$, to obtain Bayes' Rule.  
$\color{blue}p(\theta_{0.9}|x_h)= \frac{p(x_h|\theta_{0.9})p(\theta_{0.9})}{p(x_h)}$    ..................   ...............   ............. bayes's Rule




**Bayes Theorem**  
$ P(A|B) =\frac{P(A \cap B)}{P(B)}$ , this looks very simple but it has profound impact on probability theory. Let us analyze:  
[1] The left side of equation can be read as what is the probability of an event A given that event B has occured.  
[2] Numerator says, what is the probability of and event A and B occuring simultaneously.
[3] Denominator says , irrespective of any dependency on A what is the probability of occuring an event B?  
[4] Right side of equation togather tells us what fraction of B is available in $A \cap B$. bigger the portion of event B in event A, higher is the probability of $P(A|B) $
![Alt text](Bayesian1.png) fig.1  
- Suppose a person wakes up in the morning and find spots on his face. He consult a doctor and doctor told him that 90% of people who has spots on his face have smallpox, as smallpox is fatel, he is naturally terrified.  
After few moments of pondering, he asked having this symptoms what is the probability that it is smallpox and not other like chickenpox.
Ah.. doctor says, that is 1% (1 perosn in 100). of course this is not good news , but it is far better than 90%.  
- Now he start finding the data on national health portal to know what is the probability of a person having smallpox in general. He found it value 0.001.
- similarly he found that spots are observed in 80% of patient who has chickenpox.
-  $P(spots|chickenpox)=0.8$, 
    $P(spots|smallpox)=0.9$, $P(smallpox)=0.001$
- now, to find the probability $P(B)$, is via sampling of general population who have spots on their face.and therefore represents the probability that a randomly chosen individual has spots. say this value is 0.081.   All we need is  
$ P(smallpox|spots) = \frac {P(spots|smallpox) P(smallpox)} {P(spots)}$   
can be thought like this...  
$ P(Cause | Effect) = \frac {P(Effect|Cause) P(Cause)} {P(Effect)}$     
**Likelihood $P(Effect|Cause)$**  
    - The likelihood function quantifies the probability of observing the data given the parameters,It represents how well the parameters explain the observed data. 

  **Prior $P(Cause)$**  
    - A prior distribution represents an initial belief about an uncertain quantity before considering any evidence.It encapsulates subjective knowledge or beliefs before observing any data.In machine learning, a prior could be the average results that given hyperparameters produce across many machine learning projects.The prior is updated with new information to obtain the posterior probability distribution
  
**What is Bayes Theorem?**   
  [1] it describe the relationship between $P(A|B)$ and $P(B|A)$  
  [2] It expresses how a subjective degree of belief should rationally change to account for evidence.

- if we rearrange our conditional probability $P(A|B)$, then our joint probability $P(A \cap B)=P(A|B) P(B) $, In words, the joint probability of B and A is the product of the conditional probability of B, given A, and the marginal probability of A.  
**Bias-Variance trade off**  
we already know that how low bias and high variance may lead to overfitting of our model (i.e., our model is vary complicated) and lower variance and high bias may lead to underfitting (i.e., our model is too simple in estimation). let us understand this in a graphical way.  
  - when data was presented to data scientist, his job is to find out from which distribution the data has been sampled and what are its parameters, for ex. in case of Gaussian distribution data scienties need its mean and variance for the complete knowledge of distribution and then only accurate prediction can be made. This also helps to idenfify if the sample mean is the true population mean. 
  - Using this prior information he estimate the function that fits the model and can represents data well.
    ![fig 2](Bias_variance.png) fig. 2  
  - $F$ is a function class from which $f$ comes and $f$ represent the the true function that generates the data. 
  - $H$ is a function class from which we try to estimate $f$. $h_1$ is a best approximation of a $f$ in $H$. The $h_1$, $h_2$ are the estimate we get after we train our model. These $h_1$, $h_2$ estimation error or variance could be due to sample data that doesn't represent true population data or the algorithm we employed may not be proper.  
  - Bias is the difference between predicted value and expected value. i.e.,   expected approximation $ \bar f(x)$, of all samples taken  
  $\color{darkblue}\bar f(x)= E_D[h_i(x)]$  
  $\color{darkblue}Bias^2[F]= E_x[(f(x)-\bar f(x))^2]$  
  $\color{darkblue}Var[F] = E_{x,D}[(h_i(x)-\bar f(x))^2]$  
  So, if we take the expected squared deviation of a particular approximation from expected approximation we get **Variance** of a model.  
  $\color{darkgreen}MSE(F)= E_{x,D}[(f(x)- h_i(x))^2]$  
$\color{darkgreen}Bias^2(F)+Var(F)$

  **conditional independence**
  Two events A and B are conditionally independent given an event $C$ with $P(C) > 0 $ , if   
  $P(A \cap B|C)= P(A|C)P(B|C)$  
  ![Alt text](conditional_independence.png)  fig.3  
  - Let us look at the fig.3 where there could be a common elements(event) between A and B, so A can B can either be dependent or independent unconditionally. Now, think that event C has occurred, given this condiiton, we can see that A and B are independent. 
  - we can deduce that conditional probability neither implies independence nor it is implied by independence.
  - The decomposition of larget probabilistic domains into weakly connected subsets thorugh conditional independence is one of the most important developments in the recent history of AI.  
  - For example, say a person had toothache, and he visited dentist , dentist used steel probe to see if it given pain to a patient. here we've two variables, toothache and catch, if we know full joint distribution of these two variables we can get get the probability of cavity.  
  $P(Cavity|toothache \wedge catch  )=P(toothache \wedge catch| Cavity) P(Cavity)$.   
  - For this reformulation to work we need to know the conditional probabilities of conjunction $toothache \wedge catch$ for each value of cavity. This might be feasible for two evidence variables, but it does not scale up. when we have other evidence like (diet,X rays, oral hygiene, etc.), then there are $ O(2^n)$, possible combinations of observed values for which we would need to know conditional probabilities. This is no better than using the full joint distribution.
  - To make progress, we need to find some additional assertions about hte domain that will enable us to simplify the expressions. Here, the conditional independence comes to rescue.
  - Let us revisit the scenario, if probe catches in the tooth, then it is likely that the tooth has a cavity and that cavity causes toothache.   ![Alt text](Cavity.png) , Fig.3  
  - if we know cavity is present in tooth, then both toothache and catch gives not much information. i.e., they will act like an independent variables. this simply states, once we've observed cavity, then the observation probe catch doesn't provide extra information about toothache. So, when catch and toothache are dependent variables when nothing is observed, but they become independent when cavity is observed. Each is directly caused by Cavity, but neither has a direct effect on the other; toothache depends on the state of nerves in the tooth, whereas the probe's accuracy depends on dentist's skill, to which toothache is independent. Mathamatically,    
  $P(toothache \wedge catch|cavity) = P(toothache|cavity)P(catch|cavity)$
$P(toothache,catch|cavity) = P(toothache|cavity)P(catch|cavity)$
$P(X,Y |Z)= P(X|Z) P(Y|Z)$, 
