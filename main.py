import argparse
import numpy as np
import os
import torch
import torch.utils.data
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.utils as tvu
from tensorboardX import SummaryWriter


from data import *
from models import *
from utils import *


parser = argparse.ArgumentParser(description='VAE example')

# Task parametersm and model name
parser.add_argument('--uid', type=str, default='InfoVAE',
                    help='Staging identifier (default: VVAE)')

parser.add_argument('--rho', action='store_true', default=False,
                    help='Rho reparam (default: False')

# Model parameters

parser.add_argument('--z-dim', type=int, default=10, metavar='N',
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

parser.add_argument('--n_gpu', type=int, default=2)


args = parser.parse_args()

# Set cuda
use_cuda = not args.no_cuda and torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.n_gpu)


torch.manual_seed(args.seed)
if use_cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


# Data loaders
"""
Get the dataloader
"""
data_dir = 'data'
download_data = True

# do ugly if
if args.uid == 'BETAVAE' or args.uid == 'RHO_BETAVAE':
    transform = [transforms.Resize((64, 64)), transforms.ToTensor()]
else:
    transform = None

data_loader = Loader(args.dataset_name, data_dir, download_data, args.batch_size, transform, None, use_cuda)

input_shape = data_loader.img_shape
data_dim = np.prod(input_shape)
input_channels = input_shape[0]

# hack
# input_shape = data_dim
num_class = data_loader.num_class
encoder_size = args.encoder_size
decoder_size = args.encoder_size
z_dim = args.z_dim
out_channels = args.out_channels

# Set the model
model_map = {
    'VanillaVAE': VanillaVAE(data_dim, z_dim),
    'RHO_VanillaVAE': RHO_VanillaVAE(data_dim, z_dim),
    'INFO_VAE': INFO_VAE(input_shape, out_channels, encoder_size, z_dim),
    'RHO_INFO_VAE': RHO_INFO_VAE(input_shape, out_channels, encoder_size, z_dim),
    'CNN_VAE': CNN_VAE(input_channels, z_dim),
    'RHO_CNN_VAE': RHO_CNN_VAE(input_channels, z_dim),
    'BETAVAE': BETAVAE(input_channels, z_dim),
    'RHO_BETAVAE': RHO_BETAVAE(input_channels, z_dim)
}


model = model_map[args.uid].to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

# TODO init weights
model.apply(init_weights)

# Loss definition
loss_fn = loss_rho_bce_kld if args.rho else loss_bce_kld

# Set tensorboard
log_dir = args.log_dir

# dir, args.uid, timestamp
logger = SummaryWriter(comment='_' + args.uid + '_' + args.dataset_name)


def train_validate(model, loader, loss_fn, optimizer, train, use_cuda):
    model.train() if train else model.eval()
    batch_loss = 0
    batch_kld_loss = 0
    batch_bce_loss = 0
    batch_size = loader.batch_size

    for batch_idx, (x, y) in enumerate(loader):
        loss = 0
        x = to_cuda(x) if use_cuda else x
        if train:
            optimizer.zero_grad()

        if args.rho:
            x_hat, mu, rho, logs = model(x)
            loss, kld_loss, bce_loss = loss_fn(x, x_hat, mu, rho, logs, args.z_dim)
        else:
            x_hat, mu, log_var = model(x)

            loss, kld_loss, bce_loss = loss_fn(x, x_hat, mu, log_var)

        batch_loss += loss.item() / batch_size
        batch_kld_loss += kld_loss.item() / batch_size
        batch_bce_loss += bce_loss.item() / batch_size

        if train:
            loss.backward()
            optimizer.step()
    # collect better stats
    return batch_loss / (batch_idx + 1), batch_kld_loss / (batch_idx + 1), batch_bce_loss / (batch_idx + 1)


def execute_graph(model, data_loader, loss_fn, optimizer, scheduler, use_cuda):
    # Training loss
    t_loss, t_kld_loss, t_bce_loss = train_validate(model, data_loader.train_loader, loss_fn, optimizer, True, use_cuda)

    # Validation loss
    v_loss, v_kld_loss, v_bce_loss = train_validate(model, data_loader.test_loader, loss_fn, optimizer, False, use_cuda)

    print('====> Epoch: {} Average Train loss: {:.4f}'.format(epoch, t_loss))
    print('====> Epoch: {} Average KLD Train loss: {:.4f}'.format(epoch, t_kld_loss))
    print('====> Epoch: {} Average BCE Train loss: {:.4f}'.format(epoch, t_bce_loss))

    print('====> Epoch: {} Average Validation loss: {:.4f}'.format(epoch, v_loss))
    print('====> Epoch: {} Average KLD Validation loss: {:.4f}'.format(epoch, v_kld_loss))
    print('====> Epoch: {} Average BCE Validation loss: {:.4f}'.format(epoch, v_bce_loss))

    print('================================================================>')

    # Training and validation loss
    logger.add_scalar(log_dir + '/validation-loss', v_loss, epoch)
    logger.add_scalar(log_dir + '/training-loss', t_loss, epoch)

    # image generation examples
    sample = generation_example(model, args.z_dim, data_loader.train_loader, input_shape, num_class, use_cuda)
    sample = sample.detach()
    sample = tvu.make_grid(sample, normalize=False, scale_each=True)
    logger.add_image('generation example', sample, epoch)

    # image reconstruction examples
    comparison = reconstruction_example(model, args.rho, data_loader.test_loader, input_shape, num_class, use_cuda)
    comparison = comparison.detach()
    comparison = tvu.make_grid(comparison, normalize=False, scale_each=True)
    logger.add_image('reconstruction example', comparison, epoch)

    scheduler.step(v_loss)

    return v_loss


num_epochs = args.epochs
best_loss = np.inf

# Main training and validation loop
val_losses = []

for epoch in range(1, num_epochs + 1):
    v_loss = execute_graph(model, data_loader, loss_fn, optimizer, scheduler, use_cuda)
    val_losses.append(v_loss)
    if v_loss < best_loss:
        best_loss = v_loss
        print('Writing model checkpoint')
        save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'val_loss': v_loss
                        },
                        'models/' + args.uid + '_{:04.4f}.pt'.format(v_loss))
