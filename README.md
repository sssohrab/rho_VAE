# rho_VAE:

An autoregresive parametrization of the approximate posterior in VAE models.

## Main idea:

We replace the usual diagonal Gaussian parametrization of the approximate posterior in VAE models with autoregressive Gaussian.

### The standard way:

In standard VAE models, any input training sample is inducing an approximate posterior with a diagonal Gaussian distribution, i.e., <img src="/tex/7c0ac43a79518dbb936783fed0fd647e.svg?invert_in_darkmode&sanitize=true" align=middle width=221.25256725pt height=29.190975000000005pt/> to the latent space. This is realized by a linear layer that is outputing the mean vector <img src="/tex/7c2da1f3aeba73f324120131749dd5ff.svg?invert_in_darkmode&sanitize=true" align=middle width=26.561109299999988pt height=29.190975000000005pt/> of this distribution for the <img src="/tex/f802120f62e600587af32e9b7fb784d7.svg?invert_in_darkmode&sanitize=true" align=middle width=18.06055514999999pt height=27.91243950000002pt/> sample, and another linear layer that is creating the <img src="/tex/9d3a9e72dd6924b6c405e3cfb9dd8ced.svg?invert_in_darkmode&sanitize=true" align=middle width=46.354078649999984pt height=29.190975000000005pt/>, i.e., the logarithm of the diagonal elements of the covariance matrix for the <img src="/tex/f802120f62e600587af32e9b7fb784d7.svg?invert_in_darkmode&sanitize=true" align=middle width=18.06055514999999pt height=27.91243950000002pt/> sample.

Next step is to generate samples from this distribtion. The reparametrization trick is then used to generate latent codes as <img src="/tex/b6c2771b70cc8c605fdf74378709f0d5.svg?invert_in_darkmode&sanitize=true" align=middle width=158.47883699999997pt height=31.296724799999986pt/>, where <img src="/tex/9b808701e2b68072679bcc95e3891b8e.svg?invert_in_darkmode&sanitize=true" align=middle width=12.785434199999989pt height=19.1781018pt/> is the element-wise multiplication of the variance vector <img src="/tex/ef0a90f26ca7353baebee048184299fb.svg?invert_in_darkmode&sanitize=true" align=middle width=22.38150914999999pt height=29.190975000000005pt/>, with another vector of random samples generated on the fly from the zero-mean and unit-variance normal distribution. Note that this comes from the fact that the Choleskiy factorization of the diagonal covariance is another diagonal matrix whose diagonal elements are the square-root of the diagonal elements of the covariance matrix.

So in order to keep things practical, one important restriction in the choice of how to parametrize the approximate posterior distribution is to have a straightforward Choleskiy factorization of its covariance, such that it can be constructed parametrically and without having to numerically calculate it.

From the other hand, we also need to calculate the KL-divergence between this approximate posterior and the prior for the latent space, which is usually a zero-mean and unit-variance Gaussian distribution. In order to avoid many issues and further complications, we would rather have this KL-divergence term calculated in closed-form and again parametrically. This way we can easily back-propagate the gradients of the weights of the network.

With this standard diagonal Gaussian choice, this can be derived as:

<p align="center"><img src="/tex/29cb7a0e511d78491a49662219c01387.svg?invert_in_darkmode&sanitize=true" align=middle width=566.5614108pt height=32.990165999999995pt/></p>

which is very easy to implement, e.g., as ``KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())`` in PyTorch.

However, while very handy, this way of parametrization may be too simple to approximate the posterior. Note that it does not allow any correlation within the dimensions of the posterior. It just lets each dimension scale arbitrarily and independent of the other dimensions, a freedom which may even be not very necessary, since variances in the pixel domain are usually within the same range.

### The rho_VAE way:

We propose to construct the covariance matrix of the approximate posterior differently. In particular, we account for the correlation through the simplest possible way, i.e., an AR(1) process with the covariance parametrized as:

<p align="center"><img src="/tex/7a4b806fbc0d45acf099e304b2158899.svg?invert_in_darkmode&sanitize=true" align=middle width=584.9343736499999pt height=129.12876405pt/></p>
  
where <img src="/tex/82587f91ae418ac48042c24845cacbef.svg?invert_in_darkmode&sanitize=true" align=middle width=94.34343104999999pt height=21.18721440000001pt/> is a scalar value that controls the level of correlation and <img src="/tex/6f9bad7347b91ceebebd3ad7e6f6f2d1.svg?invert_in_darkmode&sanitize=true" align=middle width=7.7054801999999905pt height=14.15524440000002pt/> is another scalar that scales the whole covariance.

These two scalars can be constructed as outputs of two linear layers, each of size <img src="/tex/5b4c024baffa96a2a3f52f7733e043ac.svg?invert_in_darkmode&sanitize=true" align=middle width=41.478239549999984pt height=24.7161288pt/> (so <img src="/tex/8ecc338b886ace702c7bf09ddde6a40b.svg?invert_in_darkmode&sanitize=true" align=middle width=6.8430779999999976pt height=28.92634470000001pt/> times less parameters than the standard way for this layer). Since we want the correlation coefficient to have magnitude less than one (both positive and negative), we pass the output of the first layer through a <img src="/tex/302647a88bdde98dee6405a9b42e5340.svg?invert_in_darkmode&sanitize=true" align=middle width=40.18269584999999pt height=22.831056599999986pt/> activation. And similarly to the standard way, the output of the other layer is assumed to be the log of the positive scaling factor. 


As far as the practical issues of optimization are concerned, fortunately, this is as convenient as the standard way. The Choleskiy factorization of this covariance is parametric and has the form:

<p align="center"><img src="/tex/d8e4a33ae83a566ea947f7516bba6f3e.svg?invert_in_darkmode&sanitize=true" align=middle width=567.2374719pt height=128.2204572pt/></p>
which has structure similar to the coariance matrix itself.
  
As for the KL-divergence term, this also comes in closed-form as:

<p align="center"><img src="/tex/0588651ba3dd936f2b7c42c0efb3f6fe.svg?invert_in_darkmode&sanitize=true" align=middle width=636.8444016pt height=32.990165999999995pt/></p>
which can be implemented again very easily as:  
 
 ``KLD = 0.5 * ( torch.sum(mu.pow(2)) + - z_dim * logs - (z_dim - 1) * torch.log(1 - rho**2) +  z_dim * (logs.exp()-1)).mean() `` in PyTorch.


## A drop-in replacement:

## Some generated samples:

## Citation:
