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
  $loss_G = E_{z \in N(0,1)} l(D(G(z)), y_{real})$


> Ref: Edward Raff  
