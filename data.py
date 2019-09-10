import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms


def get_data(database_name, batch_size):
    kwargs = {'num_workers': 8, 'pin_memory': True}
    if database_name.lower() == 'mnist':
        database_obj_train = datasets.MNIST('data/mnist', train=True, download=True,
                                            transform=transforms.ToTensor())
        database_obj_test = datasets.MNIST('data/mnist', train=False, transform=transforms.ToTensor())

    elif database_name.lower() == 'cifar10':
        database_obj_train = datasets.CIFAR10('data/cifar10', train=True, download=True,
                                              transform=transforms.ToTensor())
        database_obj_test = datasets.CIFAR10('data/cifar10', train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(database_obj_train,
                                               batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(database_obj_test,
                                              batch_size=batch_size, shuffle=True, **kwargs)
    try:
        N_train, image_x, image_y, num_channel = train_loader.dataset.data.shape
        data_dim = image_x * image_y * num_channel
        input_shape = (num_channel, image_x, image_y)
    except:
        N_train, image_x, image_y = train_loader.dataset.data.shape
        data_dim = image_x * image_y
        num_channel = 1
        input_shape = (num_channel, image_x, image_y)

    return train_loader, test_loader, input_shape
