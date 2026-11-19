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
    
