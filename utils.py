import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np


# Reconstruction + KL divergence losses summed over all elements and batch
# note the change of var order!
def loss_bce_kld(x, recon_x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 1), x.view(-1, 1), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, KLD, BCE


# Note switch x, recon_x
def loss_rho_bce_kld(x, recon_x, mu, rho, logs, z_dim):

    BCE = F.binary_cross_entropy(recon_x.view(-1, 1), x.view(-1, 1), reduction='sum')
    ##
    KLD = 0.5 * (torch.sum(mu.pow(2)) + - z_dim * logs - (z_dim - 1) * torch.log(1 - rho**2) + z_dim * (logs.exp() - 1))
    KLD = torch.mean(KLD)

    return BCE + KLD, KLD, BCE


def reconstruction_example(model, use_rho, data_loader, img_shape, num_class, use_cuda):

    model.eval()
    img_shape = img_shape[1:]

    x, y = next(iter(data_loader))
    x = to_cuda(x) if use_cuda else x
    if use_rho:
        x_hat, _, _, _ = model(x)
    else:
        x_hat, _, _ = model(x)

    x = x[:num_class].cpu().view(num_class * img_shape[0], img_shape[1])
    x_hat = x_hat[:num_class].cpu().view(num_class * img_shape[0], img_shape[1])
    comparison = torch.cat((x, x_hat), 1).view(num_class * img_shape[0], 2 * img_shape[1])
    return comparison


def generation_example(model, latent_size, data_loader, img_shape, num_class, use_cuda):
    img_shape = img_shape[1:]

    draw = randn((num_class, latent_size), use_cuda)
    sample = model.decode(draw).cpu().view(num_class, 1, img_shape[0], img_shape[1])

    return sample


def to_cuda(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cuda()
    return tensor


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for sub_mod in m:
                init_weights(sub_mod)


# https://github.com/jramapuram/helpers/utils.py
def randn(shape, cuda, mean=0, sigma=1, dtype='float32'):
    shape = list(shape) if isinstance(shape, tuple) else shape
    return type_map[dtype](cuda)(*shape).normal_(mean, sigma)


def type_tdouble(use_cuda=False):
    return torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor


def type_tfloat(use_cuda=False):
    return torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


def type_tint(use_cuda=False):
    return torch.cuda.IntTensor if use_cuda else torch.IntTensor


def type_tlong(use_cuda=False):
    return torch.cuda.LongTensor if use_cuda else torch.LongTensor


def conv_size(H_in, k_size, stride, padd, dil=1):
    H_out = np.floor((H_in + 2 * padd - dil * (k_size - 1) - 1) / stride + 1)
    return np.int(H_out)


def save_checkpoint(state, filename):
    torch.save(state, filename)

# https://github.com/jramapuram/helpers/utils.py
type_map = {
    'float32': type_tfloat,
    'float64': type_tdouble,
    'double': type_tdouble,
    'int32': type_tint,
    'int64': type_tlong
}
