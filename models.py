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
        self.encoder_var = nn.Linear(encoder_size, latent_size)

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)

        mu = self.encoder_mu(x)
        var = self.encoder_var(x)
        log_var = torch.log(torch.clamp(var, min=0.01))
        return mu, log_var


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
        mu_z, log_var = self.encoder(x)
        return mu_z, log_var

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def reparameterize(self, mu_z, log_var):
        eps = torch.randn_like(log_var)
        return mu_z + eps * torch.exp(log_var)

    def forward(self, x):
        mu_z, log_var = self.encode(x)
        mu_z = self.reparameterize(mu_z, log_var)
        x_hat = self.decode(mu_z)
        return x_hat, mu_z, log_var


class RHO_DCGAN_Encoder(nn.Module):
    def __init__(self, input_shape, out_channels, encoder_size, latent_size):
        super(RHO_DCGAN_Encoder, self).__init__()

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

        self.encoder_s = nn.Linear(encoder_size, 1)
        self.encoder_rho = nn.Linear(encoder_size, 1)

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)

        mu = self.encoder_mu(x)
        rho = torch.tanh(self.encoder_rho(x))
        log_s = self.encoder_s(x)
        log_s = torch.log(torch.clamp(log_s, min=0.02))
        return mu, rho, log_s


class RHO_DCGAN_Decoder(nn.Module):
    def __init__(self, H_conv_out, out_channels, decoder_size, latent_size):
        super(RHO_DCGAN_Decoder, self).__init__()
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


class RHO_INFO_VAE(nn.Module):
    def __init__(self, input_shape, out_channels, encoder_size, latent_size):
        super(RHO_INFO_VAE, self).__init__()
        self.z_dim = latent_size
        self.encoder = RHO_DCGAN_Encoder(input_shape, out_channels, encoder_size, latent_size)
        self.decoder = RHO_DCGAN_Decoder(self.encoder.H_conv_out, out_channels, encoder_size, latent_size)

    def encode(self, x):
        mu_z, rho, log_s = self.encoder(x)
        return mu_z, rho, log_s

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def reparameterize(self, mu_z, log_s, rho):
        z_q = torch.randn_like(rho).view(-1, 1) * torch.sqrt(log_s.exp())
        for j in range(1, self.z_dim):
            addenum = z_q[:, -1].view(-1, 1) + torch.randn_like(rho).view(-1, 1) * torch.sqrt(log_s.exp())
            z_q = torch.cat((z_q, addenum), 1)
        z_q = z_q + mu_z
        return z_q

    def forward(self, x):
        mu_z, rho, log_s = self.encode(x)
        mu_z = self.reparameterize(mu_z, log_s, rho)
        x_hat = self.decode(mu_z)
        return x_hat, mu_z, rho, log_s


class CNN_VAE(nn.Module):

    def __init__(self, z_dim):
        super(CNN_VAE, self).__init__()
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 4), padding=(15, 15),
                               stride=2)  # This padding keeps the size of the image same, i.e. same padding
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), padding=(15, 15), stride=2)

        self.fc11 = nn.Linear(in_features=128 * 28 * 28, out_features=1024)
        self.fc12 = nn.Linear(in_features=1024, out_features=z_dim)

        self.fc21 = nn.Linear(in_features=128 * 28 * 28, out_features=1024)
        self.fc22 = nn.Linear(in_features=1024, out_features=z_dim)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(in_features=z_dim, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=7 * 7 * 128)
        self.conv_t1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.conv_t2 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, padding=1, stride=2)

    def encode(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = x.view(-1, 128 * 28 * 28)

        mu = F.elu(self.fc11(x))
        mu = self.fc12(mu)

        logvar = F.elu(self.fc21(x))
        logvar = self.fc22(logvar)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        x = F.elu(self.fc1(z))
        x = F.elu(self.fc2(x))
        x = x.view(-1, 128, 7, 7)
        x = F.relu(self.conv_t1(x))
        x = torch.sigmoid(self.conv_t2(x))

        return x.view(-1, 784)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar


class RHO_CNN_VAE(nn.Module):

    def __init__(self, z_dim):
        super(RHO_CNN_VAE, self).__init__()
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 4), padding=(15, 15), stride=2)  # This padding keeps the size of the image same, i.e. same padding
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), padding=(15, 15), stride=2)

        self.fc_mu_1 = nn.Linear(in_features=128 * 28 * 28, out_features=1024)
        self.fc_mu_2 = nn.Linear(in_features=1024, out_features=self.z_dim)

        self.fc_logs_1 = nn.Linear(in_features=128 * 28 * 28, out_features=1024)
        self.fc_logs_2 = nn.Linear(in_features=1024, out_features=1)

        self.fc_rho_1 = nn.Linear(in_features=128 * 28 * 28, out_features=1024)
        self.fc_rho_2 = nn.Linear(in_features=1024, out_features=1)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(in_features=self.z_dim, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=7 * 7 * 128)
        self.conv_t1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.conv_t2 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, padding=1, stride=2)

    def encode(self, x):
        # General encoder block
        x = x.view(-1, 1, 28, 28)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = x.view(-1, 128 * 28 * 28)

        mu = F.elu(self.fc_mu_1(x))
        mu = self.fc_mu_2(mu)

        log_s = F.elu(self.fc_logs_1(x))
        log_s = self.fc_logs_2(log_s)

        rho = F.elu(self.fc_rho_1(x))
        rho = torch.tanh(self.fc_rho_2(rho))

        return mu, rho, log_s

    def reparameterize(self, mu, rho, logs):

        z_q = torch.randn_like(rho).view(-1, 1) * torch.sqrt(logs.exp())
        for j in range(1, self.z_dim):
            addenum = z_q[:, -1].view(-1, 1) + torch.randn_like(rho).view(-1, 1) * torch.sqrt(logs.exp())
            z_q = torch.cat((z_q, addenum), 1)
        z_q = z_q + mu
        return z_q

    def decode(self, z):
        x = F.elu(self.fc1(z))
        x = F.elu(self.fc2(x))
        x = x.view(-1, 128, 7, 7)
        x = F.relu(self.conv_t1(x))
        x = torch.sigmoid(self.conv_t2(x))

        return x.view(-1, 784)

    def forward(self, x):
        mu, rho, log_s = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, rho, log_s)

        return self.decode(z), mu, rho, log_s
