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
