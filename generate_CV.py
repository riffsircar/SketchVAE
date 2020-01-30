import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os

import cma
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import math
import random

from PIL import Image
from scipy.spatial import distance
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from model_11 import get_model, load_model

import warnings

warnings.filterwarnings("ignore")
import corner
import dcor

char2int = {'#': 0, '-': 1, 'H': 2, 'S': 3}
int2char = {0: '#', 1: '-', 2: 'H', 3: 'S'}
num_tiles = len(char2int)
nz = 32
model = load_model('vae_cv_2900.pth',num_tiles)
model.eval()
print(model)
chunk_folder = 'chunks_CV/'
dims = (11,16)

def get_z_from_file(f):
    global model
    chunk_1 = open(chunk_folder + f, 'r').read().splitlines()
    chunk_1 = [line.replace('\r\n','') for line in chunk_1]
    out_1 = []
    for line in chunk_1:
        line_list = list(line)
        line_list_map = [char2int[x] for x in line_list]
        out_1.append(line_list_map)
    out_1 = np.asarray(out_1)
    #print(out_1, out_1.shape)
    out1_onehot = np.eye(num_tiles, dtype='uint8')[out_1]
    out1_onehot = np.rollaxis(out1_onehot, 2, 0)

    out1_onehot = out1_onehot[None, :, :]

    data_1 = torch.DoubleTensor(out1_onehot)
    #print(data_1)
    z_1, _, _ = model.encode(data_1)

    return z_1

def get_level_from_z(z):
    global model
    level = model.decode(z)
    im = level.data.cpu().numpy()
    im = np.argmax(im, axis=1).squeeze(0)
    level = np.zeros(im.shape)
    level = []
    for i in im:
        level.append(''.join([int2char[t] for t in i]))
    return level
    
def interpolate_chunks(f1, f2, num_linp=10):
    z1, z2 = get_z_from_file(f1), get_z_from_file(f2)

    alpha_values = np.linspace(0, 1, num_linp)

    vectors = []
    for alpha in alpha_values:
        vector = z1*(1-alpha) + z2*alpha
        vectors.append(vector)

    for idx, vector in enumerate(vectors):
        level = get_level_from_z(vector)
        for l in level:
            print(l)
        print('\n')

def plagiarism(z1,z2):
    l1, l2 = get_level_from_z(z1), get_level_from_z(z2)
    q, w = [], []
    rows, cols = 0, 0
    q = [list(l) for l in l1]
    w = [list(l) for l in l2]
    qt, wt = np.array(q).transpose(), np.array(w).transpose()
    for a, b in zip(q,w):
        if ''.join(a) == ''.join(b):
            rows += 1

    for a, b in zip(qt,wt):
        if ''.join(a) == ''.join(b):
            cols += 1
    return (rows+cols)
    

def density(z):
    global model
    level = get_level_from_z(z)
    total = 0
    for l in level:
        total += len(l)-l.count('-')
    return ((total*100)/(dims[0]*dims[1]))

def nonlinearity(z):
    global model
    level = get_level_from_z(z)
    level = [[level[j][i] for j in range(len(level))] for i in range(len(level[0]))]
    x = np.arange(dims[1])
    y = []
    for i, lev in enumerate(level):
        appended = False
        for j, l in enumerate(lev):
            if l != '-':
                y.append(dims[0]-j)
                appended = True
                break
        if not appended:
            y.append(0)
    x = x.reshape(-1,1)
    y = np.asarray(y)
    reg = linear_model.LinearRegression()
    reg.fit(x,y)
    y_pred = reg.predict(x)
    mse = mean_squared_error(y,y_pred)
    return mse


z1 = torch.DoubleTensor(1, nz).normal_(0,1)
z2 = torch.DoubleTensor(1, nz).normal_(0,1)
gen_d, gen_nl, tot_p = 0, 0, 0
zs = []
for _ in range(100):
    z = torch.DoubleTensor(1,nz).normal_(0,1)
    zs.append(z)

train_z = []
train_d, train_nl, num_chunks = 0, 0, 0
for file in os.listdir(chunk_folder):
    fz = get_z_from_file(file)
    train_z.append(fz)
    
closest = {"nonlin": None, "density":None, "plagiarism":None}
furthest = {"nonlin": None, "density":None, "plagiarism":None}
nonlin_closest, density_closest, plag_furthest = float('inf'), float('inf'), float('inf')
nonlin_furthest, density_furthest, plag_closest = float('-inf'), float('-inf'), float('-inf')

z_mets, t_mets = [], []

for i, tz in enumerate(train_z):
    tzd, tzn = density(tz), nonlinearity(tz)
    train_d += tzd
    train_nl += tzn
    t_mets.append([tzd,tzn])
    

for i, z in enumerate(zs):
    print(i)
    d, nl = density(z),nonlinearity(z)
    gen_d += d
    gen_nl += nl
    z_mets.append([d,nl])
    for j, tm in enumerate(t_mets):
        tzd, tzn = tm[0], tm[1]
        tz = train_z[j]
        den_delta, nl_delta = math.fabs(d - tzd), math.fabs(nl - tzn)
        p = plagiarism(z,tz)
        tot_p += p
        if den_delta < density_closest:
            density_closest = den_delta
            closest['density'] = (z,tz,density_closest)
        if den_delta > density_furthest:
            density_furthest = den_delta
            furthest['density'] = (z,tz,density_furthest)

        if nl_delta < nonlin_closest:
            nonlin_closest = nl_delta
            closest['nonlin'] = (z,tz,nonlin_closest)
        if nl_delta > nonlin_furthest:
            nonlin_furthest = nl_delta
            furthest['nonlin'] = (z,tz,nonlin_furthest)

        if p > plag_closest:
            plag_closest = p
            closest['plagiarism'] = (z,tz,plag_closest)
        if p < plag_furthest:
            plag_furthest = p
            furthest['plagiarism'] = (z,tz,plag_furthest)
        


z, tz, val = closest['density']
print("DENSITY CLOSEST: ", val)
print("Generated: ")
level = get_level_from_z(z)
for l in level:
    print(l)
print('\n')
print("Training: ")
level = get_level_from_z(tz)
for l in level:
    print(l)
print('\n\n')

z, tz, val = furthest['density']
print("DENSITY FURTHEST: ", val)
print("Generated: ")
level = get_level_from_z(z)
for l in level:
    print(l)
print('\n')
print("Training: ")
level = get_level_from_z(tz)
for l in level:
    print(l)
print('\n\n')

z, tz, val = closest['nonlin']
print("NONLIN CLOSEST: ", val)
print("Generated: ")
level = get_level_from_z(z)
for l in level:
    print(l)
print('\n')
print("Training: ")
level = get_level_from_z(tz)
for l in level:
    print(l)
print('\n\n')

z, tz, val = furthest['nonlin']
print("NONLIN FURTHEST: ", val)
print("Generated: ")
level = get_level_from_z(z)
for l in level:
    print(l)
print('\n')
print("Training: ")
level = get_level_from_z(tz)
for l in level:
    print(l)
print('\n\n')

z, tz, val = closest['plagiarism']
print("PLAGIARISM CLOSEST: ", val)
print("Generated: ")
level = get_level_from_z(z)
for l in level:
    print(l)
print('\n')
print("Training: ")
level = get_level_from_z(tz)
for l in level:
    print(l)
print('\n\n')

z, tz, val = furthest['plagiarism']
print("PLAGIARISM FURTHEST: ", val)
print("Generated: ")
level = get_level_from_z(z)
for l in level:
    print(l)
print('\n')
print("Training: ")
level = get_level_from_z(tz)
for l in level:
    print(l)
print('\n\n')
print(len(zs), len(train_z))
print("Avg Train Density: ", train_d/len(train_z))
print("Avg Gen Density: ", gen_d/len(zs))
print("Avg Train NL: ", train_nl/len(train_z))
print("Avg Gen NL: ", gen_nl/len(zs))
print("Avg Plagiarism: ", tot_p/(len(train_z)*len(zs)))
print("ED: ", dcor.energy_distance(z_mets,t_mets))
sys.exit()
    

z = get_z_from_file('CV_chunk_100.txt')
print(nonlinearity(z))
    
for _ in range(10):
    z = torch.DoubleTensor(1, nz).normal_(0,1)
    level = get_level_from_z(z)
    for l in level:
        print(l)

    print('\n')

interpolate_chunks('CV_chunk_100.txt','CV_chunk_1000.txt')
z = get_z_from_file('smb_chunk_150.txt')
level = get_level_from_z(z)
for l in level:
    print(l)
