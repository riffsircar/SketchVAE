from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import sys
from PIL import Image
from model_11 import get_model, load_model
from torch.utils.data import DataLoader, Dataset, TensorDataset

device = torch.device('cpu')
latent_dim = 32 # size of latent vector
batch_size = 32 # input batch size
folder = 'chunks_CV/'

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

def parse_folder(folder):
    levels = []
    text = ''
    for file in os.listdir(folder):
        with open(os.path.join(folder,file),'r') as infile:
            level = []
            for line in infile:
                text += line
                level.append(list(line.rstrip()))
            
            levels.append(level)
    return levels, text

levels, text = parse_folder(folder)
text = text.replace('\n','')
print(len(levels))
print("Num batches: ", len(levels)/batch_size)
chars = sorted(list(set(text.strip('\n'))))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
print(char2int)
num_tiles = len(char2int)
print(num_tiles)

encoded = []
for level in levels:
    enc = []
    for line in level:
        encoded_line = [char2int[x] for x in line]
        enc.append(encoded_line)
    encoded.append(enc)
encoded = np.array(encoded)
print(encoded.shape)
print(encoded[0])

onehot = np.eye(num_tiles, dtype='uint8')[encoded]
onehot = np.rollaxis(onehot, 3, 1)
print(onehot.shape)

train = torch.from_numpy(onehot).to(dtype=torch.float64)
train_ds = TensorDataset(train,train)
train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True)

vae, opt = get_model(torch.device('cpu'), num_tiles, latent_dim)
print(vae)
#sys.exit()

def loss_fn(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')/recon_x.size(0)
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

epochs = 10000 # num epochs to train for
out_file = open('cv_loss.csv','w')
out_file.write('Train Loss\n')

for i in range(epochs):
    vae.train()
    train_loss = 0
    kld_loss = 0
    for batch, (x,_) in enumerate(train_dl):
        x = x.to(device)
        opt.zero_grad()
        recon_x, mu, logvar = vae(x)
        loss, bce, kld = loss_fn(recon_x, x, mu, logvar)
        train_loss += loss.item()
        kld_loss += kld.item()
        loss.backward()
        opt.step()
    train_loss /= len(train_dl.dataset)
    kld_loss /= len(train_dl.dataset)
    print('Epoch: ', i,'\tLoss: ',train_loss,"\tKLD: ", kld_loss)
    if i % 100 == 0:
        print('Epoch: {} Loss: {:.4f}'.format(i, train_loss))
        out_file.write(str(train_loss) + '\n')
        torch.save(vae.state_dict(), 'vae_cv_' + str(i) + '.pth')
torch.save(vae.state_dict(), 'vae_cv_final.pth')
out_file.close()
