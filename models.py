import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F


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
