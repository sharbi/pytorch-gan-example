from __future__ import print_function

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
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
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 5000
lr = 0.0002
beta = 0.5
ngpu = 1
# Create dataset

class SvhnDataset(Dataset):
    def __init__(self, image_size, split):
        self.split = split
        self.use_gpu = True if torch.cuda.is_available() else False

        self.svhn_dataset = self._create_dataset(image_size, split)
        self.label_mask = self._create_label_mask()

    def _create_dataset(self, image_size, split):
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize])
        return dset.SVHN(root='./svhn', download=True, transform=transform, split=split)

    def _is_train_dataset(self):
        return True if self.split == 'train' else False

    def _create_label_mask(self):
        if self._is_train_dataset():
            label_mask = np.zeros(len(self.svhn_dataset))
            label_mask[0:1000] = 1
            np.random.shuffle(label_mask)
            label_mask = torch.LongTensor(label_mask)
            return label_mask
        return None

    def __len__(self):
        return len(self.svhn_dataset)

    def __getitem__(self, idx):
        data, label = self.svhn_dataset.__getitem__(idx)
        if self._is_train_dataset():
            return data, label, self.label_mask[idx]
        return data, label

def get_loader(image_size, batch_size):
    num_workers = 2

    svhn_train = SvhnDataset(image_size=image_size, split='train')
    svhn_test = SvhnDataset(image_size=image_size, split='test')

    svhn_loader_train = DataLoader(
        dataset=svhn_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    svhn_loader_test = DataLoader(
        dataset=svhn_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return svhn_loader_train, svhn_loader_test

svhn_loader_train, _ = get_loader(image_size=image_size, batch_size=batch_size)
image_iter = iter(svhn_loader_train)
images, _, _ = image_iter.next()


device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Sort weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def _to_var(x):
    if ngpu > 0:
        x = x.cuda()
    return Variable(x)


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

        max_val, _ = torch.max(real_class_logits, 1, keepdim=True)
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

        features = self.features(out)
        features = features.squeeze()

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
        label_onehot = _to_var(torch.FloatTensor(label_onehot))
        return label_onehot

netG = Generator(ngpu).to(device)
netG.apply(weights_init)
print(netG)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
print(netD)

d_criterion = nn.BCEWithLogitsLoss()
d_gan_criterion = nn.CrossEntropyLoss()
fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)
fixed_noise = _to_var(fixed_noise)


d_gan_labels_real = torch.FloatTensor(batch_size)
d_gan_labels_fake = torch.FloatTensor(batch_size)

real_label = 1
fake_label = 0


optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta, 0.999))

schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=[500, 1000, 1500, 2000, 2500, 2750], gamma=0.1)
schedulerG = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=[500, 1000, 1500, 2000, 2500, 2750], gamma=0.1)


for epoch in range(num_epochs):

    masked_correct = 0
    num_samples = 0
    schedulerD.step()
    schedulerG.step()
    # For each batch in the dataloader
    for i, data in enumerate(svhn_loader_train):
        svhn_data, svhn_labels, label_mask = data
        svhn_data = _to_var(svhn_data)
        svhn_labels = _to_var(svhn_labels).long().squeeze()
        label_mask = _to_var(label_mask).float().squeeze()


        ##########################
        # FIRST SORT OUT SUPERVISED LOSS:
        # This is checking how well the discriminator can categorise the real data
        ##########################


        netD.zero_grad()
        d_gan_labels_real = d_gan_labels_real.resize_as_(svhn_labels.data.cpu().float()).uniform_(0, 0.3)
        d_gan_labels_real_var = _to_var(d_gan_labels_real).float()
        output, d_class_logits_on_data, gan_logits_real, d_sample_features = netD(svhn_data)

        svhn_labels_one_hot = one_hot(svhn_labels)
        d_class_loss_entropy = d_criterion(torch.log(output), svhn_labels_one_hot)

        d_class_loss_entropy = d_class_loss_entropy.squeeze()
        delim = torch.max(torch.Tensor([1.0, torch.sum(label_mask.data)]))
        delim = _to_var(delim)
        supervised_loss = torch.sum(label_mask * d_class_loss_entropy) / delim

        ##########################
        # NEXT UNSUPERVISED LOSS:
        # This checks the discriminator's ability to determine real and fake data
        ##########################

        # Get the fake logits, real are obtained from above


        noise = torch.FloatTensor(batch_size, nz, 1, 1)

        noise.resize_(svhn_labels.data.shape[0], nz, 1, 1).normal_(0, 1)
        noise_var = _to_var(noise)
        fake = netG(noise_var)

        d_gan_labels_fake = d_gan_labels_fake.resize_as_(svhn_labels.data.cpu().float()).uniform_(0.9, 1.2)
        d_gan_labels_fake_var = _to_var(d_gan_labels_fake).float()
        _, _, gan_logits_fake, _ = netD(fake.detach())


        real_unsupervised_loss = d_criterion(
            gan_logits_real,
            d_gan_labels_real_var
        )

        fake_unsupervised_loss = d_criterion(
            gan_logits_fake,
            d_gan_labels_real_var
        )

        unsupervised_loss = real_unsupervised_loss + fake_unsupervised_loss

        loss_d = supervised_loss + unsupervised_loss

        loss_d.backward(retain_graph=True)
        optimizerD.step()

        ##########################
        # Now train the Generator
        # This is based on the feature differences between real and fake
        ##########################

        netG.zero_grad()


        noise = torch.FloatTensor(batch_size, nz, 1, 1)
        noise.resize_(svhn_labels.data.shape[0], nz, 1, 1).normal_(0, 1)
        noise_var = _to_var(noise)
        # Generate fake images
        fake = netG(noise_var)

        _, _, _, d_data_features = netD(fake)

        data_features_mean = torch.mean(d_data_features, dim=0).squeeze()
        sample_features_mean = torch.mean(d_sample_features, dim=0).squeeze()

        g_loss = torch.mean(torch.abs(data_features_mean - sample_features_mean))

        g_loss.backward()
        optimizerG.step()

        _, pred_class = torch.max(d_class_logits_on_data, 1)
        eq = torch.eq(svhn_labels, pred_class)
        correct = torch.sum(eq.float())
        masked_correct += torch.sum(label_mask * eq.float())
        num_samples += torch.sum(label_mask)

        if i % 200 == 0:
            print('Training:\tepoch {}/{}\tdiscr. gan loss {}\tdiscr. class loss {}\tgen loss {}\tsamples {}/{}'.
                  format(epoch, num_epochs,
                         unsupervised_loss.data[0], supervised_loss.data[0],
                         g_loss.data[0], i + 1,
                         len(svhn_loader_train)))
            real_cpu, _, _ = data
            vutils.save_image(real_cpu,
                    './.gitignore/output/SS_GAN_TEST/real_samples.png',
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                    './.gitignore/output/SS_GAN_TEST/fake_samples_epoch_%03d.png' % epoch,
                    normalize=True)

    accuracy = masked_correct.data[0]/max(1.0, num_samples.data[0])
    print('Training:\tepoch {}/{}\taccuracy {}'.format(epoch, num_epochs, accuracy))
