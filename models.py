import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from utils import conv_size


class BatchFlatten(nn.Module):
    def __init__(self):
        super(BatchFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class BatchReshape(nn.Module):
    def __init__(self, shape):
        super(BatchReshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()


class ConvReluBatch(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(ConvReluBatch, self).__init__(*args, **kwargs)
        batch_dim = self.weight.data.size(0)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(batch_dim)

    def forward(self, x):
        x = super(ConvReluBatch, self).forward(x)
        return self.bn(self.relu(x))


class ConvTrReluBatch(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super(ConvTrReluBatch, self).__init__(*args, **kwargs)
        batch_dim = self.weight.data.size(1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(batch_dim)

    def forward(self, x):
        x = super(ConvTrReluBatch, self).forward(x)
        return self.bn(self.relu(x))


class VanillaVAE(nn.Module):
    def __init__(self, data_dim, z_dim):
        super(VanillaVAE, self).__init__()
        self.data_dim = data_dim
        self.z_dim = z_dim

        self.fc1 = nn.Linear(self.data_dim, 400)
        self.fc21 = nn.Linear(400, self.z_dim)
        self.fc22 = nn.Linear(400, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, 400)
        self.fc4 = nn.Linear(400, self.data_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.data_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class RhoVanillaVAE(nn.Module):
    def __init__(self, data_dim, z_dim):
        super(RhoVanillaVAE, self).__init__()
        self.data_dim = data_dim
        self.z_dim = z_dim

        self.fc1 = nn.Linear(self.data_dim, 400)
        self.fc21 = nn.Linear(400, self.z_dim)
        self.fc22 = nn.Linear(400, 1)  # rho
        self.fc23 = nn.Linear(400, 1)  # s
        self.fc3 = nn.Linear(self.z_dim, 400)
        self.fc4 = nn.Linear(400, self.data_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), torch.tanh(self.fc22(h1)), self.fc23(h1)    # -1<rho<1,and s>0

    def reparameterize(self, mu, rho, logs):

        z_q = torch.randn_like(rho).view(-1, 1) * torch.sqrt(logs.exp())
        for j in range(1, self.z_dim):
            addenum = z_q[:, -1].view(-1, 1) + torch.randn_like(rho).view(-1, 1) * torch.sqrt(logs.exp())
            z_q = torch.cat((z_q, addenum), 1)
        z_q = z_q + mu
        return z_q

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, rho, logs = self.encode(x.view(-1, self.data_dim))
        z = self.reparameterize(mu, rho, logs)
        return self.decode(z), mu, rho, logs


class DCGAN_Encoder(nn.Module):
    def __init__(self, input_shape, out_channels, encoder_size, latent_size):
        super(DCGAN_Encoder, self).__init__()

        H_conv_out = conv_size(input_shape[-1], 4, 2, 1)
        H_conv_out = conv_size(H_conv_out, 3, 1, 1)
        H_conv_out = conv_size(H_conv_out, 4, 2, 1)
        H_conv_out = conv_size(H_conv_out, 3, 1, 1)

        convnet_out = np.int(H_conv_out * H_conv_out * out_channels * 2)

        self.H_conv_out = H_conv_out

        self.encoder = nn.ModuleList([
            # in_channels, out_channels, kernel_size, stride=1, padding=0
            ConvReluBatch(1, out_channels, 4, 2, padding=1),

            ConvReluBatch(out_channels, out_channels, 3, 1, padding=1),
            ConvReluBatch(out_channels, out_channels * 2, 4, 2, padding=1),
            ConvReluBatch(out_channels * 2, out_channels * 2, 3, 1, padding=1),

            BatchFlatten(),
            nn.Linear(convnet_out, encoder_size)
        ])

        self.encoder_mu = nn.Linear(encoder_size, latent_size)
        self.encoder_std = nn.Linear(encoder_size, latent_size)

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)

        mu = self.encoder_mu(x)
        std = self.encoder_std(x)
        std = torch.clamp(torch.sigmoid(std), min=0.01)
        return mu, std


class DCGAN_Decoder(nn.Module):
    def __init__(self, H_conv_out, out_channels, decoder_size, latent_size):
        super(DCGAN_Decoder, self).__init__()
        self.decoder = nn.ModuleList([
            nn.Linear(latent_size, decoder_size),
            nn.ReLU(),

            nn.Linear(decoder_size, H_conv_out * H_conv_out * out_channels * 2),
            nn.ReLU(),
            BatchReshape((out_channels * 2, H_conv_out, H_conv_out, )),

            ConvTrReluBatch(out_channels * 2, out_channels, 4, 2, padding=1),
            ConvTrReluBatch(out_channels, out_channels, 3, 1, padding=1),
            ConvTrReluBatch(out_channels, out_channels // 2, 4, 2, padding=1),

            nn.ConvTranspose2d(out_channels // 2, 1, 3, 1, padding=1),
            nn.Sigmoid()
        ])

    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x


class INFO_VAE(nn.Module):
    def __init__(self, input_shape, out_channels, encoder_size, latent_size):
        super(INFO_VAE, self).__init__()
        self.encoder = DCGAN_Encoder(input_shape, out_channels, encoder_size, latent_size)
        self.decoder = DCGAN_Decoder(self.encoder.H_conv_out, out_channels, encoder_size, latent_size)

    def encode(self, x):
        mu_z, std_z = self.encoder(x)
        return mu_z, std_z

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def reparameterize(self, mu_z, std_z):
        eps = torch.randn_like(std_z)
        return mu_z + eps * std_z

    def forward(self, x):
        mu_z, std_z = self.encode(x)
        mu_z = self.reparameterize(mu_z, std_z)
        x_hat = self.decode(mu_z)
        return x_hat, mu_z, std_z
