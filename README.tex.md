# rho_VAE:

An autoregresive parametrization of the approximate posterior in VAE models.

## Main idea:

We replace the usual diagonal Gaussian parametrization of the approximate posterior in VAE models with autoregressive Gaussian.

### Standard way:

In standard VAE models, any input training sample is inducing an approximate posterior with a diagonal Gaussian distribution, i.e., $q_{\phi}(\mathbf{z}|\mathbf{x}^{(i)}) = \mathcal{N} \big( \mathbf{\mu}^{(i)}, \text{diag} (\mathbf{s}^{(i)})  \big)$ to the latent space. This is realized by a linear layer that is outputing the mean vector $\boldsymbol{\mu}^{(i)}$ of this distribution for the $i^{\text{th}}$ sample, and another linear layer that is creating the $\log{\mathbf{s}^{(i)}}$, i.e., the logarithm of the diagonal elements of the covariance matrix for the $i^{\text{th}}$ sample.

Next step is to generate samples from this distribtion. The reparametrization trick is then used to generate latent codes as $\mathbf{z}^{(i)} = \boldsymbol{\mu}^{(i)} + \sqrt{\mathbf{s}^{(i)}} \odot \boldsymbol{\epsilon}$, where $\odot$ is the element-wise multiplication of the variance vector $\mathbf{s}^{(i)}$, with another vector of random samples generated on the fly from the zero-mean and unit-variance normal distribution. Note that this comes from the fact that the Choleskiy factorization of the diagonal covariance is another diagonal matrix whose diagonal elements are the square-root of the diagonal elements of the covariance matrix.

So in order to keep things practical, one important restriction in the choice of how to parametrize the approximate posterior distribution is to have a straightforward Choleskiy factorization of its covariance, such that it can be constructed parametrically and without having to numerically calculate it.

From the other hand, we also need to calculate the KL-divergence between this approximate posterior and the prior for the latent space, which is usually a zero-mean and unit-variance Gaussian distribution. In order to avoid many issues and further complications, we would rather have this KL-divergence term calculated in closed-form and again parametrically. This way we can easily back-propagate the gradients of the weights of the network.

With this standard diagonal Gaussian choice, this can be derived as:

$$ D_{\text{KL}}\Big[ \mathcal{N} \Big( \boldsymbol{\mu}^{(i)}, \text{diag} \big( \mathbf{s}^{(i)} \big) \Big)    \Big|\Big|  \mathcal{N} \big( \mathbf{0}, \mathrm{I}_d \big)  \Big] = \frac{1}{2} \Big[ \mathbf{1}_d^T \mathbf{s}^{(i)} + \big|\big| \boldsymbol{\mu}^{(i)} \big|\big|_2^2 -d -  \mathbf{1}_d^T \log \big({\mathbf{s}^{(i)}} \big) \Big],$$

which is very easy to implement, e.g., as ``KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())`` in PyTorch.

### rho_VAE way:

