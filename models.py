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
