# rho_VAE:

An autoregresive parametrization of the approximate posterior in VAE models.

## Main idea:

We replace the usual diagonal Gaussian parametrization of the approximate posterior in VAE models with autoregressive Gaussian.

### Standard way:

In standard VAE models, any input training sample is inducing an approximate posterior with a diagonal Gaussian distribution, i.e., <img src="/tex/4b0e2a39976428930788cda4d27da88a.svg?invert_in_darkmode&sanitize=true" align=middle width=219.52131465000002pt height=29.190975000000005pt/> to the latent space. This is realized by a linear layer that is outputing the mean vector <img src="/tex/7c2da1f3aeba73f324120131749dd5ff.svg?invert_in_darkmode&sanitize=true" align=middle width=26.561109299999988pt height=29.190975000000005pt/> of this distribution for the <img src="/tex/f802120f62e600587af32e9b7fb784d7.svg?invert_in_darkmode&sanitize=true" align=middle width=18.06055514999999pt height=27.91243950000002pt/> sample, and another linear layer that is creating the <img src="/tex/9d3a9e72dd6924b6c405e3cfb9dd8ced.svg?invert_in_darkmode&sanitize=true" align=middle width=46.354078649999984pt height=29.190975000000005pt/>, i.e., the logarithm of the diagonal elements of the covariance matrix for the <img src="/tex/f802120f62e600587af32e9b7fb784d7.svg?invert_in_darkmode&sanitize=true" align=middle width=18.06055514999999pt height=27.91243950000002pt/> sample.

Next step is to generate samples from this distribtion. The reparametrization trick is then used to generate latent codes as <img src="/tex/b6c2771b70cc8c605fdf74378709f0d5.svg?invert_in_darkmode&sanitize=true" align=middle width=158.47883699999997pt height=31.296724799999986pt/>, where <img src="/tex/9b808701e2b68072679bcc95e3891b8e.svg?invert_in_darkmode&sanitize=true" align=middle width=12.785434199999989pt height=19.1781018pt/> is the element-wise multiplication of the variance vector <img src="/tex/ef0a90f26ca7353baebee048184299fb.svg?invert_in_darkmode&sanitize=true" align=middle width=22.38150914999999pt height=29.190975000000005pt/>, with another vector of random samples generated on the fly from the zero-mean and unit-variance normal distribution. Note that this comes from the fact that the Choleskiy factorization of the diagonal covariance is another diagonal matrix whose diagonal elements are the square-root of the diagonal elements of the covariance matrix.

So in order to keep things practical, one important restriction in the choice of how to parametrize the approximate posterior distribution is to have a straightforward Choleskiy factorization of its covariance, such that it can be constructed parametrically and without having to numerically calculate it.

From the other hand, we also need to calculate the KL-divergence between this approximate posterior and the prior for the latent space, which is usually a zero-mean and unit-variance Gaussian distribution. In order to avoid many issues and further complications, we would rather have this KL-divergence term calculated in closed-form and again parametrically. This way we can easily back-propagate the gradients of the weights of the network.

With this standard diagonal Gaussian choice, this can be derived as:

<p align="center"><img src="/tex/29cb7a0e511d78491a49662219c01387.svg?invert_in_darkmode&sanitize=true" align=middle width=566.5614108pt height=32.990165999999995pt/></p>

which is very easy to implement, e.g., as ``KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())`` in PyTorch.

### rho_VAE way:

