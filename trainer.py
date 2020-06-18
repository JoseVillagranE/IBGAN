# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 23:42:56 2020

@author: joser
"""

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence

import torchvision.datasets as Dataset
from torchvision import utils as vutils

import tqdm
from itertools import chain
from os.path import join as pjoin
import numpy as np

from models import IBGAN
from torchvision import transforms
import torch.utils.data as data

def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def trans_maker(size=256):
	trans = transforms.Compose([transforms.Resize((size+10, size+10)),
					transforms.CenterCrop((size, size)), 
					#transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					#_rescale
					])
	return trans

def KL_Loss(z):
	mu = z.mean()
	logvar = z.var().log()
	return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())




def train(Model, args_dict):
    print("-------- training --------")
    dataset = Dataset.ImageFolder(root=args_dict['DATA_ROOT'], transform=trans_maker(64)) 
    dataloader = iter(DataLoader(dataset, args_dict['BATCH_SIZE'], sampler=InfiniteSamplerWrapper(dataset), num_workers=0, pin_memory=True))
    
    loss_bce = nn.BCELoss()
    loss_mse = nn.MSELoss()
    
    M_r = MultivariateNormal(loc=torch.zeros(args_dict['R_DIM']).to(args_dict['device']), 
    scale_tril=torch.ones(args_dict['R_DIM'], args_dict['R_DIM']).to(args_dict['device']))
    
    D_real = D_fake = D_z_kl = G_real = Z_recon = R_kl = 0
    fixed_z = torch.randn(64, args_dict['Z_DIM']).to(args_dict['device'])
    
    LOG_INTERVAL = args_dict['LOG_INTERVAL']
    
    for n_iter in tqdm.tqdm(range(0, args_dict['N_EPOCHS'])):
        
        real_image = next(dataloader)[0].to(args_dict['device'])
        z = torch.randn(args_dict['BATCH_SIZE'], args_dict['Z_DIM']).to(args_dict['device'])
        r_sampler = Model.r_sampler(z)
        g_image = Model.generate(r_sampler.sample())
        
        Model.OptD.zero_grad()
        Model.OptQ.zero_grad()
        pred_f = Model.discriminate(g_image.detach())
        pred_r, rec_z = Model(real_image)
        d_loss = (loss_bce(torch.sigmoid(pred_r), torch.ones(pred_r.size()).to(args_dict['device'])) + 
                loss_bce(torch.sigmoid(pred_f), torch.zeros(pred_f.size()).to(args_dict['device'])))
        q_loss = KL_Loss(rec_z)
        #d_loss.backward()
        total_loss = d_loss + q_loss
        total_loss.backward()
        Model.OptD.step()

        D_real += torch.sigmoid(pred_r).mean().item()
        D_fake += torch.sigmoid(pred_f).mean().item()
        D_z_kl += q_loss.item()
        
        Model.OptD.zero_grad()
        Model.OptG.zero_grad()

        pred_g, z_posterior = Model(g_image)
        
        g_loss = args_dict['LAMBDA_G']* loss_bce(torch.sigmoid(pred_g), torch.ones(pred_g.size()).to(args_dict['device']))
        # reconstruction loss of z
        ## TODO
        ## question here: as stated in the paper-algorithm-1: this part should be a - log(q(z|x)) instead of mse
        recon_loss = loss_mse(z_posterior, z)
        # kl loss between e(r|z) || m(r) as a variational inference
        #kl_loss = BETA_KL * torch.distributions.kl.kl_divergence(r_likelihood, M_r).mean()
        kl_loss = args_dict['BETA_KL']*kl_divergence(r_sampler, M_r).mean()
        total_loss = g_loss + recon_loss + kl_loss
        total_loss.backward()
        Model.OptE.step()
        Model.OptG.step()

        # record the loss values
        G_real += torch.sigmoid(pred_g).mean().item()
        Z_recon += recon_loss.item()
        R_kl += kl_loss.item()
        
        if n_iter % args_dict['LOG_INTERVAL'] == 0 and n_iter > 0:
            print("D(x): %.5f    D(G(z)): %.5f    D_kl: %.5f    G(z): %.5f    Z_rec: %.5f    R_kl: %.5f"% (D_real/LOG_INTERVAL, D_fake/LOG_INTERVAL, D_z_kl/LOG_INTERVAL, G_real/LOG_INTERVAL, Z_recon/LOG_INTERVAL, R_kl/LOG_INTERVAL))
            D_real = D_fake = D_z_kl = G_real = Z_recon = R_kl = 0
        

if __name__ == "__main__":
    BATCH_SIZE = 128
    Z_DIM = 500
    R_DIM = 15
    NDF = 64
    NGF = 64
    N_EPOCHS = 100000
    LOG_INTERVAL = 1

    LAMBDA_G = 1
    BETA_KL = 0.3
    DATA_ROOT = "img_align_celeba/"
    device = "cpu"
    args_dict = {"BATCH_SIZE": BATCH_SIZE, "Z_DIM": Z_DIM, "R_DIM": R_DIM, "NDF": NDF, "NGF": NGF, "N_EPOCHS": N_EPOCHS, 
             "LAMBDA_G": LAMBDA_G, "BETA_KL": BETA_KL, "device": device, "DATA_ROOT": DATA_ROOT, "LOG_INTERVAL": LOG_INTERVAL}
        
    lr = 1e-5
    IBGAN_model = IBGAN(NGF, NDF, Z_DIM, R_DIM, lr, lr, lr, lr)
    IBGAN_model = IBGAN_model.to(device)
    
    train(IBGAN_model, args_dict)