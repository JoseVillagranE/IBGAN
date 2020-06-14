# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 21:24:26 2020

@author: joser
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter, init
from torch.nn.utils import spectral_norm
from torch.distributions.multivariate_normal import MultivariateNormal
from itertools import chain

class UnFlatten(nn.Module):
    
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        return x.view(x.size(0), -1, self.block_size, self.block_size)
        
class Flatten(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)

class IBGAN(nn.Module):
    
    
    def __init__(self, ngf, ndf, z_dim, r_dim, lr_G, lr_D, lr_Q, lr_E, nc=3):
        
        super().__init__()
        
        self.r_dim = r_dim
        
        # Block programing with respect of paper
        
        self.Block_E = nn.Sequential(nn.Linear(z_dim, ngf*2), nn.BatchNorm1d(ngf*2), nn.ReLU(),
                                     nn.Linear(ngf*2, ngf), nn.BatchNorm1d(ngf), nn.ReLU(),
                                     nn.Linear(ngf, r_dim*2))
        
        self.Block_G = nn.Sequential(nn.Linear(r_dim, ngf*16), nn.BatchNorm1d(ngf*16), nn.ReLU(),
                                     nn.Linear(ngf*16, 64*4*ngf), nn.BatchNorm1d(64*4*ngf), nn.ReLU(),
                                     UnFlatten(4),
                                     #nn.Conv2d(ngf*4, ngf*4, 3), nn.BatchNorm2d(ngf*4), nn.ReLU(),
                                     #nn.Conv2d(ngf*4, ngf*4, 3), nn.BatchNorm2d(ngf*4), nn.ReLU(),
                                     nn.ConvTranspose2d(ngf*4, ngf*2, 4), nn.BatchNorm2d(ngf*2), nn.ReLU(),
                                     nn.ConvTranspose2d(ngf*2, ngf, 4), nn.BatchNorm2d(ngf), nn.ReLU(),
                                     nn.ConvTranspose2d(ngf, nc), nn.Tanh())
        
        self.SubBlock_QD = nn.Sequential(nn.Conv2d(nc, ndf, 4), nn.ReLU(),
                                         nn.Conv2d(ndf, ndf*2), nn.BatchNorm2d(ndf*2), nn.ReLU(),
                                         nn.Conv2d(ndf*2, ndf*4, 4), nn.BatchNorm2d(ndf*4), nn.ReLU(),
                                         nn.Conv2d(ndf*4, ndf*16, 8), nn.BatchNorm2d(16*ndf), nn.ReLU(),
                                         )
       
        self.Block_Q = nn.Sequential(Flatten(),
                                     nn.Linear(ndf*64*16, ndf*16), nn.BatchNorm2d(ndf*16), nn.ReLU(),
                                     nn.Linear(ndf*16, z_dim))
        
        self.Block_D = spectral_norm(nn.Conv2d(ndf*16, 1, 4))
        
        self.OptD = optim.RMSprop(chain[self.Block_D.parameters(), self.SubBlock_QD.parameters()], 
                                  lr=lr_D, momentum=0.9)
        self.OptG = optim.RMSprop(self.Block_G.parameters(), lr=lr_G, momentum=0.9)
        self.OptE = optim.RMSprop(self.Block_E.parameters(), lr=lr_E, momentum=0.9)
        self.OptQ = optim.RMSprop(chain[self.Block_Q.parameters(), self.SubBlock_QD.parameters()], 
                                  lr=lr_Q, momentum=0.9)
        
    def r_sampler(self, x):
        r = self.Block_E(x)
        mu = r[:, :self.r_dim]
        var = F.softplus(r[:, self.r_dim:]) + 1e-5
        scale_tri = torch.diag_embed(var)
        return MultivariateNormal(loc=mu, scale_tril=scale_tri)
    
    def generate(self, x):
        g = self.Block_G(x)
        return g
        
    def forward(self, x):
        
        x = self.SubBlock_QD(x)
        Q = self.Block_Q(x)
        D = self.Block_D(x)
        return D.view(-1), Q
    
    