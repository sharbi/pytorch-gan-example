from __future__ import print_function
#%matplotlib inline

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

manual_seed = 123

random.seed(manual_seed)
torch.manual_seed(manual_seed)

# Set initial paramaters
batch_size = 128
image_size = 64
nc = 1
nz = 100
ngf = 64
ndf = 64
num_epochs = 200
lr = 0.0002
beta = 0.5
ngpu = 1

class Discriminator(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, )
