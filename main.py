import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import torchvision.utils as tvu
from tensorboardX import SummaryWriter
import matplotlib as mpl
import matplotlib.pyplot as plt

from data import get_data
from models import VanillaVAE, RhoVanillaVAE, INFO_VAE, RHO_INFO_VAE

from utils import loss_bce_kld, loss_rho_bce_kld, to_cuda, init_weights, reconstruction_example, generation_example

mpl.use('Agg')

parser = argparse.ArgumentParser(description='VAE example')

# Task parameters
parser.add_argument('--uid', type=str, default='VVAE',
                    help='Staging identifier (default: Vanilla VAE)')

# Model parameters
parser.add_argument('--rho', action='store_true', default=False,
                    help='Rho reparameterization (default: True')

parser.add_argument('--z-dim', type=int, default=5, metavar='N',
                    help='VAE latent size (default: 20')

parser.add_argument('--out-channels', type=int, default=64, metavar='N',
                    help='VAE 2D conv channel output (default: 64')

parser.add_argument('--encoder-size', type=int, default=1024, metavar='N',
                    help='VAE encoder size (default: 1024')


# data loader parameters
parser.add_argument('--dataset-name', type=str, default='mnist',
                    help='Name of dataset (default: MNIST')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input training batch-size')


# Optimizer
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of training epochs')

parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate (default: 1e-4')


# Log directory
parser.add_argument('--log-dir', type=str, default='runs',
                    help='logging directory (default: runs)')


# Device (GPU)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')

parser.add_argument('--seed', type=int, default=1,
                    help='Seed for numpy and pytorch (default: None')


args = parser.parse_args()

# Set cuda
use_cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if use_cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# Data loaders
train_loader, test_loader, input_shape = get_data(args.dataset_name, args.batch_size)
data_dim = np.prod(input_shape)

# hack
# input_shape = data_dim
num_class = 10
encoder_size = args.encoder_size
decoder_size = args.encoder_size
latent_size = args.z_dim
out_channels = args.out_channels

# Model def
if args.rho:
    model = RHO_INFO_VAE(input_shape, out_channels, encoder_size, latent_size).to(device)
else:
    model = INFO_VAE(input_shape, out_channels, encoder_size, latent_size).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# TODO init weights
model.apply(init_weights)

# Loss definition
loss_fn = loss_rho_bce_kld if args.rho else loss_bce_kld

# Set tensorboard
log_dir = args.log_dir
logger = SummaryWriter()


def train_validate(model, loader, loss_fn, optimizer, train, use_cuda):
    model.train() if train else model.eval()
    batch_loss = 0
    batch_size = loader.batch_size

    for batch_idx, (x, y) in enumerate(loader):
        loss = 0
        x = to_cuda(x) if use_cuda else x
        if train:
            optimizer.zero_grad()

        if args.rho:
            x_hat, mu, rho, logs = model(x)
            loss = loss_fn(x, x_hat, mu, rho, logs, args.z_dim, data_dim)
        else:
            x_hat, mu, log_var = model(x)
            loss = loss_fn(x, x_hat, mu, log_var, data_dim)
        batch_loss += loss.item() / batch_size

        if train:
            loss.backward()
            optimizer.step()
    # collect better stats
    return batch_loss / (batch_idx + 1)


def execute_graph(model, train_loader, test_loader, loss_fn, optimizer, use_cuda):
    # Training loss
    t_loss = train_validate(model, train_loader, loss_fn, optimizer, True, use_cuda)

    # Validation loss
    v_loss = train_validate(model, test_loader, loss_fn, optimizer, False, use_cuda)

    print('====> Epoch: {} Average Train loss: {:.4f}'.format(
          epoch, t_loss))
    print('====> Epoch: {} Average Validation loss: {:.4f}'.format(
          epoch, v_loss))

    # Training and validation loss
    logger.add_scalar(log_dir + '/validation-loss', v_loss, epoch)
    logger.add_scalar(log_dir + '/training-loss', t_loss, epoch)

    # image generation examples
    sample = generation_example(model, args.z_dim, train_loader, input_shape, use_cuda)
    sample = sample.detach()
    sample = tvu.make_grid(sample, normalize=False, scale_each=True)
    logger.add_image('generation example', sample, epoch)

    # image reconstruction examples
    comparison = reconstruction_example(model, args.rho, test_loader, input_shape, use_cuda)
    comparison = comparison.detach()
    comparison = tvu.make_grid(comparison, normalize=False, scale_each=True)
    logger.add_image('reconstruction example', comparison, epoch)

    return v_loss


num_epochs = args.epochs

# Main training and validation loop
for epoch in range(1, num_epochs + 1):
    _ = execute_graph(model, train_loader, test_loader, loss_fn, optimizer, use_cuda)
