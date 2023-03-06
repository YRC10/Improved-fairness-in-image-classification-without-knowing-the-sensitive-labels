import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from torchvision import datasets, transforms
import torchvision.transforms as tfs
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

os.system('pip install pretrainedmodels')
import pretrainedmodels as ptm


class Densenet121(nn.Module):
    def __init__(self, pretrained):
        super(Densenet121, self).__init__()
        self.model = ptm.__dict__["densenet121"](
            num_classes=1000, pretrained="imagenet" if pretrained else None
        )

    def forward(self, x):
        x = self.model.features(x)
        # print(x.shape)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        return x
    


# from torchvision.models import densenet121

class bvaeEncoder(nn.Module):
    def __init__(self, latent_size):
        super(bvaeEncoder, self).__init__()
        self.encoder = Densenet121(pretrained=True)

        self.feature_dim = self.encoder.model.last_linear.in_features

        self.fc = nn.Linear(self.feature_dim, latent_size)


        # self.densenet = densenet121(pretrained=True)
        # self.latent_size = latent_size
        # self.features = nn.Sequential(*list(self.densenet.children())[:-1])
        self.fc_mu = nn.Linear(self.feature_dim, latent_size)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_size)

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        # x = self.fc(x)
        
        # x = self.features(x)
        # x = F.relu(x, inplace=True)
        # x = nn.AdaptiveAvgPool2d((1, 1))(x)
        # x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    # def reparameterize(self, mu, logvar):
    #     std = logvar.mul(0.5).exp_()
    #     eps = torch.randn_like(std)
    #     return eps.mul(std).add_(mu)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
class bvaeDecoder(nn.Module):
    def __init__(self, latent_size):
        super(bvaeDecoder, self).__init__()
        self.latent_size = latent_size
        self.fc = nn.Linear(latent_size, 1024)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 1024, 1, 1)
        x = self.deconv(x)
        return x


class BetaVAE(nn.Module):
    def __init__(self, latent_size, beta):
        super(BetaVAE, self).__init__()
        self.encoder = bvaeEncoder(latent_size)
        self.decoder = bvaeDecoder(latent_size)
        self.latent_size = latent_size
        self.beta = beta

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

    def loss_function(self, x_recon, x, mu, logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kl_loss
        return loss