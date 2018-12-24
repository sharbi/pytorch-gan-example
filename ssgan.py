from __future__ import print_function

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

manual_seed = 123

# Set random seeds
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# Set initial paramaters
workers = 2
batch_size = 128
image_size = 32
num_classes = 10
nc = 1
nz = 100
ngf = 64
ndf = 64
num_epochs = 200
lr = 0.0002
beta = 0.5
ngpu = 1
# Create dataset

os.makedirs('./.gitignore/data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    dset.MNIST('./.gitignore/data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(image_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=batch_size, shuffle=True)


device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Sort weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):

    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is Z, going into convolution
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        return self.main(input)

class _ganLogits(nn.Module):

    def __init__(self):
        super(_ganLogits, self).__init__()


    def forward(self, class_logits):
        real_class_logits, fake_class_logits = torch.split(class_logits, num_classes, dim=1)
        fake_class_logits = torch.squeeze(fake_class_logits)

        max_val, _ = torch.max(real_class_logits, -1, keepdim=True)
        stable_class_logits = real_class_logits - max_val
        max_val = torch.squeeze(max_val)
        gan_logits = torch.log(torch.sum(torch.exp(stable_class_logits), 1)) + max_val - fake_class_logits

        return gan_logits


class Discriminator(nn.Module):

    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(
            nn.Dropout2d(0.5/2.5),

            # input is (number_channels) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.5),
            # (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2),
            # (ndf) x 8 x 8
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.5),
            # (ndf) x 4 x 4
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),
            # (ndf * 2) x 4 x 4
            nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),
            # (ndf * 2) x 4 x 4
            nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 0, bias=False),
            nn.LeakyReLU(0.2),
            # (ndf * 2) x 2 x 2
        )

        self.features = nn.AvgPool2d(kernel_size=2)

        self.class_logits = nn.Linear(
            in_features=(ndf * 2) * 1 * 1,
            out_features=num_classes + 1)

        self.gan_logits = _ganLogits()

        self.softmax = nn.Softmax(dim=0)

    def forward(self, inputs):

        out = self.main(inputs)

        print(out.size())

        features = self.features(out)
        features = features.squeeze()

        print(features.size())

        class_logits = self.class_logits(features)

        gan_logits = self.gan_logits(class_logits)

        out = self.softmax(class_logits)

        return out, class_logits, gan_logits, features


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def one_hot(x):
        label_numpy = x.data.cpu().numpy()
        label_onehot = np.zeros((label_numpy.shape[0], num_classes + 1))
        label_onehot[np.arange(label_numpy.shape[0]), label_numpy] = 1
        label_onehot = torch.FloatTensor(label_onehot).to(device)
        return label_onehot

netG = Generator(ngpu).to(device)
netG.apply(weights_init)
print(netG)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
print(netD)

d_criterion = nn.BCEWithLogitsLoss()
fixed_noise = torch.randn(64, nz, 1, 1, device=device)


real_label = 1
fake_label = 0


optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta, 0.999))

for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ##########################
        # FIRST SORT OUT SUPERVISED LOSS:
        # This is checking how well the discriminator can categorise the real data
        ##########################

        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        real_labels = torch.full((b_size, ), real_label, device=device)
        output, _, gan_logits_real, d_sample_features = netD(real_cpu)

        mnist_labels = one_hot(data[1])
        supervised_loss = torch.mean(d_criterion(mnist_labels, torch.log(output)))

        ##########################
        # NEXT UNSUPERVISED LOSS:
        # This checks the discriminator's ability to determine real and fake data
        ##########################

        # Get the fake logits, real are obtained from above


        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake images
        fake = netG(noise)

        _, _, gan_logits_fake, _ = netD(fake.detach())

        logits_sum_real = torch.logsumexp(gan_logits_real, 1)
        logits_sum_fake = torch.logsumexp(gan_logits_fake, 1)

        unsupervised_loss = 0.5 * (
            -(torch.mean(logits_sum_real)) +
            torch.mean(torch.nn.Softplus(logits_sum_real)) +
            torch.mean(torch.nn.Softplus(logits_sum_fake))
        )

        loss_d = supervised_loss + unsupervised_loss

        loss_d.backward(retain_graph=True)
        optimizerD.step()

        ##########################
        # Now train the Generator
        # This is based on the feature differences between real and fake
        ##########################


        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake images
        fake = netG(noise)
        fake_labels = torch.full((b_size, ), fake_label, device=device)

        _, _, _, d_data_features = netD(fake)

        data_features_mean = torch.mean(d_data_features, dim=0).squeeze()
        sample_features_mean = torch.mean(d_sample_features, dim=0).squeeze()

        g_loss = torch.mean(torch.abs(data_features_mean - sample_features_mean))

        g_loss.backward()
        self.g_optimizer.step()

        if i % 200 == 0:
            print('Training:\tepoch {}/{}\tdiscr. gan loss {}\tdiscr. class loss {}\tgen loss {}\tsamples {}/{}'.
                  format(epoch, self.opt.epochs,
                         d_gan_loss.data[0], d_class_loss.data[0],
                         g_loss.data[0], i + 1,
                         len(self.svhn_loader_train)))
            real_cpu = data
            vutils.save_image(real_cpu,
                              '{}/real_samples.png'.format(self.opt.out_dir),
                              normalize=True)
            fake = self.netG(fixed_noise)
            vutils.save_image(fake.data,
                              '{}/fake_samples_epoch_{:03d}.png'.format(self.opt.out_dir, epoch),
                              normalize=True)
