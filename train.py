#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:13:26 2022

@author: giannis_pitsiorlas
"""

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
import torch


import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
from IPython.display import clear_output

from utils import *
from wgan import *


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

""" So generally both torch.Tensor and torch.cuda.Tensor are equivalent. You can do everything you like with them both.
The key difference is just that torch.Tensor occupies CPU memory while torch.cuda.Tensor occupies GPU memory.
Of course operations on a CPU Tensor are computed with CPU while operations for the GPU / CUDA Tensor are computed on GPU."""

##############################################
# Defining all hyperparameters
##############################################


class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


hp = Hyperparameters(
    n_epochs=400,
    batch_size=64,
    lr=0.0005,
    n_cpu=8,
    latent_dim=64,
    img_size=28,
    channels=3,
    n_critic=25,
    clip_value=0.001,
    sample_interval=800,
)

##############################################
# Setting Root Path for Google Drive or Kaggle
##############################################
root_path = "PixelPeps/data"

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
)

train_dataset = torchvision.datasets.ImageFolder(root=root_path, transform = transforms)
loader = torch.utils.data.DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True)



##############################################
# INITIALIZE Generator and Critic
##############################################

img_shape = (hp.channels, hp.img_size, hp.img_size)

generator = Generator(img_shape, hp.latent_dim)
critic = Critic(img_shape)

if cuda:
    generator.cuda()
    critic.cuda()

##############################################
# Defining all Optimizers
##############################################

optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=hp.lr)
optimizer_D = torch.optim.RMSprop(critic.parameters(), lr=hp.lr)


##############################################
# Initialize weights
##############################################
generator.apply(weights_init_normal)
critic.apply(weights_init_normal)


def train():
    for epoch in range(hp.n_epochs):
        for i, (imgs, _) in enumerate(loader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            #########################
            #  Train Critic
            #########################

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], hp.latent_dim))))

            # Generate a batch of images
            # fake_imgs = generator(z).detach()
            fake_imgs = generator(img_shape, z).detach()

            """ The math for the loss functions for the critic and generator is:
                Critic Loss: D(x) - D(G(z))
                -D(x) + D(G(z))
                Generator Loss: D(G(z))
                Now for the Critic Loss, as per the Paper, we have to maximize the expression.
                So, arithmetically, maximizing an expression, means minimizing the -ve of that expression
                i.e. -(D(x) - D(G(z))) which is -D(x) + D(G(z)) i.e. -D(real_imgs) + D(G(real_imgs))
            """
            d_loss = -torch.mean(critic(real_imgs)) + torch.mean(critic(fake_imgs))

            d_loss.backward()
            optimizer_D.step()

            """ Clip weights of critic to avoid vanishing/exploding gradients in the
            critic/critic. """
            for p in critic.parameters():
                p.data.clamp_(-hp.clip_value, hp.clip_value)

            """ In WGAN, we update Critic more than Generator
            Train the generator every n_critic iterations
            we need to increase training iterations of the critic so that it works to
            approximate the real distribution sooner.
            """
            if i % hp.n_critic == 0:
                #########################
                #  Train Generators
                #########################
                optimizer_G.zero_grad()

                # Generate a batch of images
                fake_images_from_generator = generator(img_shape, z)
                # Adversarial loss
                g_loss = -torch.mean(critic(fake_images_from_generator))

                g_loss.backward()
                optimizer_G.step()

            ##################
            #  Log Progress
            ##################

            batches_done = epoch * len(loader) + i

            if batches_done % hp.sample_interval == 0:
                clear_output()
                print(f"Epoch:{epoch}:It{i}:DLoss{d_loss.item()}:GLoss{g_loss.item()}")
                visualise_output(fake_images_from_generator.data[:50], 10, 10)


train()