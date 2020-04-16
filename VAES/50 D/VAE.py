from __future__ import print_function
import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.utils.data
import torch.nn.init as init
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class VAE(nn.Module):

    def __init__(self, x_dim, h_dim, z_dim):

        super(VAE, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        #encoder
        self.enc = nn.Sequential( nn.Linear(self.x_dim , self.h_dim), nn.Tanh())
        self.enc_mean = nn.Sequential( nn.Linear(self.h_dim, self.z_dim) )
        self.enc_std = nn.Sequential( nn.Linear(self.h_dim, self.z_dim))

        #decoder
        self.dec = nn.Sequential( nn.Linear(self.z_dim , self.h_dim), nn.Tanh())
        self.dec_mean = nn.Sequential( nn.Linear( self.h_dim, self.x_dim ))
        self.dec_std = nn.Sequential( nn.Linear(self.h_dim, self.x_dim))
        

    def encode (self, x ):
        #print (x.shape)
        enc = self.enc(x.float())
        enc_mean = self.enc_mean(enc)
        enc_std = self.enc_std(enc)
        return enc_mean , enc_std


    def decode (self, z):
        dec = self.dec(z)
        dec_mean = self.dec_mean(dec)
        dec_std = self.dec_std(dec)
        return dec_mean , dec_std

    def forward(self, x):
        kld_loss = 0
        nll_loss = 0
        #encoder
        enc_mean , enc_std = self.encode(x)
        #sampling and reparameterization
        z = self._reparameterized_sample(enc_mean, enc_std)
        #decoder
        dec_mean , dec_std = self.decode(z)
        kld_loss += self._kld_gauss(enc_mean, enc_std.mul(0.5).exp_())
        nll_loss += self._nll_gauss(dec_mean, dec_std, x)
        return kld_loss, nll_loss,(enc_mean , enc_std),(dec_mean , dec_std) , z


    def _reparameterized_sample(self, mean, logvar):
        """using std to sample"""
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)


    def _kld_gauss(self, mu, logvar):
        """Using std to compute KLD"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def _nll_gauss(self, mean, logvar , x):
        return torch.sum( 0.5 * np.log (2 * np.pi) + 0.5 * logvar + (x-mean)**2 / (2 *  torch.exp(logvar)) )
