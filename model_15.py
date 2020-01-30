import torch.nn as nn
import torch.optim
from torch.nn import DataParallel
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        #print(input.shape)
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input):
        #print(input.shape)
        return input.view(input.size(0), 512, 1, 1)

class Shape(nn.Module):
    def forward(self, input):
        print(input.shape)
        return input

class VAE(nn.Module):
    def __init__(self, nc=2, z_dim=32, h_dim=512):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            #input shape: n, 4, 15, 16
            #Shape(),
            nn.Conv2d(nc, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            #Shape(),
            #output shape: n, 64, 6, 7

            #input shape: n, 64, 6, 7
            nn.Conv2d(64,128,kernel_size=4,stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            #output shape: n, 128, 2, 2
            #Shape(),
            #input shape: n, 128, 2, 2
            Flatten()
            #output shape: n, 128 * 2 * 2 = 512
        )
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            #Shape(),
            #input: n, 512, 1, 1
            nn.ConvTranspose2d(h_dim, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #output: n, 128, 4, 4
            
            #Shape(),
            #input: n, 128, 4, 4
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #output: n, 64, 7, 7 
            #Shape(),

            #input: n, 64, 7, 7
            nn.ConvTranspose2d(64, nc, kernel_size=4, stride=1),
            #Shape(),
            #output: n, 2, 10, 10
            nn.Upsample((15,16)),
            #Shape(),
            nn.Sigmoid()
        )

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        esp = esp.to(dtype=torch.float64)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
    
def get_model(dev, nc, z_dim, lr=1e-3):
    model = VAE(nc, z_dim)
    model = model.to(dev).double()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return model, opt

def load_model(path, nc, dev=torch.device('cpu')):
    model = VAE(nc).double().to(dev)
    model.load_state_dict(torch.load(path, map_location=dev))
    return model
