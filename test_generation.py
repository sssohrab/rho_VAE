import argparse
import numpy as np
import os
import torch.utils.data
from torchvision.utils import save_image

from data import *
from models import *
from utils import *


parser = argparse.ArgumentParser(description='VAE example')

# Task parametersm and model name
parser.add_argument('--uid', type=str, default='InfoVAE',
                    help='Staging identifier (default: VVAE)')

parser.add_argument('--rho', action='store_true', default=False,
                    help='Rho reparam (default: False')

parser.add_argument('--z-dim', type=int, default=10, metavar='N',
                    help='VAE latent size (default: 10')

parser.add_argument('--out-channels', type=int, default=64, metavar='N',
                    help='VAE 2D conv channel output (default: 64')

parser.add_argument('--encoder-size', type=int, default=1024, metavar='N',
                    help='VAE encoder size (default: 1024')

# data loader parameters
parser.add_argument('--dataset-name', type=str, default='mnist',
                    help='Name of dataset (default: MNIST')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input training batch-size')

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


print("Running {} on dataset {}".format(args.uid, args.dataset_name))

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
loaded_model = torch.load('models/' + args.uid + args.dataset_name + '.pt')
model.load_state_dict(loaded_model['state_dict'])
sample = generation_example(model, args.z_dim, data_loader.train_loader, input_shape, num_class - 2, use_cuda)
sample = sample.detach()
save_image(sample, 'output/generation_{}_{}.png'.format(args.uid, args.dataset_name))
