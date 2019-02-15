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
import pandas as pd

import pickle as pkl

manual_seed = 123

# Set random seeds
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# Set initial paramaters
workers = 2
batch_size = 100
image_size = 60
num_classes = 2
nc = 1
nz = 100
ngf = 64
ndf = 64
num_epochs = 5000
lr = 0.0003
beta = 0.5
ngpu = 1
# Create dataset

class DiabetesDataset(Dataset):
    def __init__(self, root_dir, data_file, transform, split):
        self.split = split
        self.root_dir = root_dir
        self.use_gpu = True if torch.cuda.is_available() else False
        self.transform = transform
        self.diabetes_dataset = pd.read_csv(root_dir + data_file)
        self.diabetes_dataset = self.diabetes_dataset.to_numpy()


    def _is_train_dataset(self):
        return True if self.split == 'train' else False


    def __len__(self):
        return len(self.diabetes_dataset)

    def __getitem__(self, idx):
        data = self.diabetes_dataset.__getitem__(idx)

        labels = data[6]
        data = data[1:6]


        data = np.expand_dims(data, axis=0)
        data = np.expand_dims(data, axis=0)

        if self._is_train_dataset():
            return data, data, labels
        return data, labels

def get_loader(batch_size):
    num_workers = 2

    normalize = transforms.Normalize(
        mean=[0.5, 0.5],
        std=[0.5, 0.5])
    transform = transforms.Compose([
        transforms.ToTensor()])

    diabetes_train = DiabetesDataset('../diabetes_data/', 'normalised_diabetes_dataset.csv', transform=transform, split='train')
    diabetes_test = DiabetesDataset('../diabetes_data/', 'normalised_diabetes_dataset.csv', transform=transform, split='test')

    diabetes_loader_train = DataLoader(
        dataset=diabetes_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    diabetes_loader_test = DataLoader(
        dataset=diabetes_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return diabetes_loader_train, diabetes_loader_test

diabetes_loader_train, diabetes_loader_test = get_loader(batch_size=batch_size)
patient_iter = iter(diabetes_loader_train)
patient, _, _ = patient_iter.next()

test_iter = iter(diabetes_loader_test)
test_patient, _ = test_iter.next()



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
            nn.ConvTranspose2d(nz, ngf * 4, 2, (1, 2), 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 4),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 2, (1, 3), 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 2),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 2, ngf, 1, (1, 2), 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf),
            # state size. (ngf*4) x 8 x 8
            nn.utils.weight_norm(nn.ConvTranspose2d(ngf, nc, 1, (1, 1), 0, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Tanh()
        # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        out = self.main(input)
        return out

class _ganLogits(nn.Module):

    def __init__(self):
        super(_ganLogits, self).__init__()


    def forward(self, class_logits):
        real_class_logits, fake_class_logits = torch.split(class_logits, num_classes, dim=1)
        fake_class_logits = torch.squeeze(fake_class_logits)

        max_val, _ = torch.max(real_class_logits, 1, keepdim=True)
        stable_class_logits = real_class_logits - max_val
        max_val = torch.squeeze(max_val)
        gan_logits = torch.logsumexp(stable_class_logits, 1) + max_val - fake_class_logits

        return gan_logits



class Discriminator(nn.Module):

    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(

            nn.Dropout(0.2),

            # input is (number_channels) x 60 x 4
            nn.utils.weight_norm(nn.Conv2d(nc, ndf, 3, padding=1, bias=False)),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv2d(ndf, ndf, 3, padding=1, bias=False)),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv2d(ndf, ndf, 3, padding=1, stride=2, bias=False)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            # (ndf) x 30 x 2
            nn.utils.weight_norm(nn.Conv2d(ndf, ndf *2, 3, padding=1, bias=False)),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv2d(ndf*2, ndf*2, (1, 2), padding=1, bias=False)),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv2d(ndf*2, ndf*2, 3, padding=1, stride=2, bias=False)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            # (ndf) x 15 x 1
            nn.utils.weight_norm(nn.Conv2d(ndf*2, ndf*2, 1, padding=0, bias=False)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        )

        self.features = nn.AvgPool2d(kernel_size=2)

        self.class_logits = nn.Linear(
            in_features=(ndf * 2) * 1 * 1,
            out_features=num_classes)

        #self.gan_logits = _ganLogits()

        #self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, inputs):

        out = self.main(inputs)

        features = self.features(out)
        features = features.squeeze()

        class_logits = self.class_logits(features)

        #gan_logits = self.gan_logits(class_logits)

        #out = self.softmax(class_logits)

        return class_logits, features


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
print(netG)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
print(netD)


d_unsupervised_criterion = nn.BCEWithLogitsLoss()
d_gan_criterion = nn.CrossEntropyLoss()
fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)
fixed_noise = _to_var(fixed_noise)


d_gan_labels_real = torch.FloatTensor(batch_size)
d_gan_labels_fake = torch.FloatTensor(batch_size)

real_label = 1
fake_label = 0


optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta, 0.999))


for epoch in range(num_epochs):

    masked_correct = 0
    num_samples = 0
    # For each batch in the dataloader
    for i, data in enumerate(diabetes_loader_train):
        labeled_data, unlabeled_data, labels = data
        labeled_data = _to_var(labeled_data).float()
        unlabeled_data = _to_var(unlabeled_data).float()
        labels = _to_var(labels).long().squeeze()

        noise = torch.FloatTensor(batch_size, nz, 1, 1)

        noise.resize_(labels.data.shape[0], nz, 1, 1).uniform_(0, 100)
        noise_var = _to_var(noise)
        generator_input = netG(noise_var)

        pert_input = noise.resize_(labels.data.shape[0], nz, 1, 1).normal_(0, 100)
        pert_n = F.normalize(pert_input)

        noise_pert = noise + 1. * pert_n
        noise_pert = _to_var(noise_pert)
        gen_inp_pert = netG(noise_pert)



        manifold_regularisation_value = (gen_inp_pert - generator_input)
        manifold_regularisation_norm = F.normalize(manifold_regularisation_value)

        gen_adv = generator_input + 20. * manifold_regularisation_norm



        ##########################
        # FIRST SORT OUT SUPERVISED LOSS:
        # This is checking how well the discriminator can categorise the real data
        ##########################


        netD.zero_grad()
        logits_lab, _ = netD(labeled_data)
        logits_unl, layer_real = netD(unlabeled_data)
        logits_gen, _ = netD(generator_input.detach())
        logits_gen_adv, _ = netD(gen_adv.detach())

        print(logits_lab.shape)
        print(labels.shape)

        l_unl = torch.logsumexp(logits_unl, 1)
        l_gen = torch.logsumexp(logits_gen, 1)
        loss_lab = d_gan_criterion(logits_lab, labels)
        loss_lab = torch.mean(loss_lab)

        loss_unlabeled = - 0.5 * torch.mean() \
                         + 0.5 * torch.mean(F.softplus(l_unl)) \
                         + 0.5 * torch.mean(F.softplus(l_gen))


        manifold = torch.sum(torch.sqrt(torch.square(logits_gen - logits_gen_adv) + 1e-8))

        j_loss = torch.mean(manifold)

        loss_d = loss_unl + loss_lab + (0.001 * j_loss)


        loss_d.backward(retain_graph=True)
        optimizerD.step()

        ##########################
        # Now train the Generator
        # This is based on the feature differences between real and fake
        ##########################

        netG.zero_grad()

        _, layer_fake = netD(fake)


        m1 = torch.mean(layer_real, dim=0).squeeze()
        m2 = torch.mean(layer_fake, dim=0).squeeze()


        loss_g = torch.mean(torch.abs(m1 - m2))


        loss_g.backward()
        optimizerG.step()

        if i % 200 == 0:
            print('Training:\tepoch {}/{}\tdiscr. gan loss {}\tdiscr. class loss {}\tgen loss {}\tsamples {}/{}'.
                  format(epoch, num_epochs,
                         loss_unl.data[0], loss_lab.data[0],
                         loss_g.data[0], i + 1,
                         len(diabetes_loader_train)))
            real_cpu, _, _ = data
            vutils.save_image(real_cpu,
                    './.gitignore/output/SS_GAN_TEST/real_samples.png',
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                    './.gitignore/output/SS_GAN_TEST/fake_samples_epoch_%03d.png' % epoch,
                    normalize=True)

    with torch.no_grad():
        for i, data in enumerate(diabetes_loader_test):
            for data in diabetes_loader_test:
                test_values, test_labels = data
                disc_test = netD(test_values)
                pred_class, _  = torch.max(disc_test, 1)
                correct_pred = torch.equal(torch.cast(torch.argmax(pred_class, 1),
                    torch.int32), torch.cast(test_labels, torch.int32))
                accuracy = torch.mean(torch.cast(correct_pred, torch.float32))

    accuracy = masked_correct.data[0]/max(1.0, num_samples.data[0])
    print('Training:\tepoch {}/{}\taccuracy {}'.format(epoch, num_epochs, accuracy))
