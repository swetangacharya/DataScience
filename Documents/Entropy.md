**Entropy**
    - Entropy is a measure of uncertainty: it says how much additional information would be needed to pin down the actual outcome. It does not tell you what the outcome is; it tells you how many possibilities remain compatible with what you already know.  
    $H(X)=-\sum_i p(x_i)log p(x_i)$ .  Shannon Entropy has base 2, one can still select base e or base 10.

    - Entropy depends on your state of knowledge, not merely on the object itself. If I tell you I am “in the United States,” your uncertainty about my location is high; if I give you a ZIP code, it drops sharply—even though my actual location has not changed.    
    - given your description of a system, how much detail is still unresolved? It quantifies ignorance relative to a chosen set of possible states and a chosen level of description—not absolute ignorance about everything.   

**What Entropy does not measure?**    
    
    - Which particular outcome will occur, Information you already possess, The value, importance, or meaning of the outcome.   
**What Entropy does measure**   
    
    - How many outcomes are still possible, weighted by likelihood, Information still missing, Average surprise or description length. 
### What is information at first place?    
    - information is that which allows you to make predictions with accuracy better than chance. We all understand that information is useful. But when we get some input/news about some event and we say it is not useful.  then what you were exposed to was most likely not information. That stuff, instead, was mostly Entropy. It is just a word we use to quantify how much is not known.  

**Example of a die:-**  What we do not know about the 6 faces die system is which side is going to be up when I throw it next. What you do know is that it has six sides. How much do you not know about this die? The answer is not six. This is because information/lack of information is not defined in terms of the number of unknown states. Rather, it is given by the logarithm of the number of unknown states.   
Let us quantify your uncertainty (that is, how much you do not know) about a system (system one) by the number of states it can be in. Say this is N1. Imagine that there is another system (system two), and that one can be in N2 different states. How many states can the joint system (system one and two combined) be in? Well, for each state of system one, there can be N2 number of states. which is $log(N_1)+log(N_2)$   
**True aspect of Entropy:-**   We say that the entropy of the die is whatever it is, and does not depend on the state of the observer? Well, then you are mistaken. Physical objects do not have an intrinsic uncertainty.
For example if we say we have fair die that has an entropy $log6$ and fair coin has entropy $log2$. But if we play a different game, where we count face in tail of head is in which of four quadrant.
then my entropy is 1 bit for head/tail and 2 bits for knowing which one of four quadrant it is in. which is $log_28=3$ bits. Such that we can make many such parts and keep changing the entropy.   
    So, As this fact dawns on us, we are clear that the entropy of a physical object is not defined unless you tell which degrees of freedom are important to you. In other words, it is defined by the number of states that can be resolved by the measurement that you are going to be using to determine the state of the physical object. If it is heads or tails that counts for you, then $log(2)$ is your uncertainity.If you play the ‘four-quadrant’ game, the entropy of the coin is $log(8)$, and so on.    
    Now, If I tell you that on this die $6$ won't come. then your uncertainty is reduced to $log_2(6)-log_2(5)=0.263$, this mean you gain $0.263$ bits of information gain.   
    - How the entropy of a physical system (such as a die, a coin or a book) depends on the measurement device that you will use for querying that system. That, come to think of it, the uncertainty (or entropy) of any physical object really is infinite, and made finite only by the finiteness of our measurement devices. Of course, the things you could possibly know about any physical object are infinite!   
    - Now, Let us change the definition bit. Suppose you have urn with some Red and Blue balls. How much we don't know about urn is quantified by the 'entropy' of the urn.
That is calculated from the fraction of blue balls known to be in the urn, and the fraction of red balls in the urn. You know these fractions that are common knowledge. So, let us say that the fraction of blue is $p$. The fraction of red then is of course (you do the math) $1−p$. And the entropy of the urn is  
$H(X)=-(p log(p)+ (1-p) log(1-p))$  
    - If we know nothing else about this urn, then the uncertainty is equal to the log of the number of possible states, as I wrote before. Because there are only red and blue balls, that would be log 2. If the base of the log is 2, then the result is $log_2(2)=1$ bit.  
So, if there are red and blue balls only in an urn, then I can predict the outcome of an experiment (pulling a ball from the urn) just as well as I can predict whether a fair coin lands on heads or tails. If I correctly predict the outcome (I will be able to do this about half the time, on average) I am correct purely by chance. Information is that which allows you to make a correct prediction with accuracy better than chance, which in this case means more than half of the time.   

**Entropy and Cross-Entropy**
Entropy tells us how much uncertainity actually exist in reality. No model involved yet. 
$H(P) = -\sum p(x) log p(x)$   

**Cross-Entropy** tells us how many bits we would need if reality is P but we encode/predict using model Q.
$H(P,Q)= - \sum p(x) logq(x)$
Important to notice that reality hasn't changed, it is still P. The only thing that has changed is our description of reality.   
**Cross-entropy**= the apparent uncertainty when we use our model Q to describe a world that actually follows P.

**KL Divergence:-** It does not measure uncertainity. it measures The extra uncertainty (or extra coding cost) caused by using the wrong beliefs. It is the panelty. It is the gap. It's avoidable part.
    
