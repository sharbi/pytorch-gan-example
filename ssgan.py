from __future__ import print_function

import argparse
import os
import random
import math
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
batch_size = 256
image_size = 60
num_classes = 2
nc = 1
nz = 100
ngf = 64
ndf = 64
num_epochs = 5000
lr = 0.0003
beta = 0.5
ngpu = 0
labeled_rate = 0.03
# Create dataset

class DiabetesDataset(Dataset):
    def __init__(self, root_dir, data_file, split, labeled):
        self.split = split
        self.root_dir = root_dir
        self.labeled = labeled
        self.use_gpu = True if torch.cuda.is_available() else False
        self.diabetes_dataset = pd.read_csv(root_dir + data_file)
        self.diabetes_dataset = self.diabetes_dataset.values



    def _is_train_dataset(self):
        return True if self.split == 'train' else False


    def __len__(self):
        return len(self.diabetes_dataset)

    def __getitem__(self, idx):
        data = self.diabetes_dataset.__getitem__(idx)

        labels = []

        if not self._is_train_dataset:
            labels = data[:, 6]
            data = data[:, 0:5]


        if self.labeled:
            self.labels = pd.read_csv('../diabetes_data/labels.csv')

        data = np.expand_dims(data, axis=0)
        data = np.expand_dims(data, axis=0)


        if self.labeled or self._is_train_dataset:
            return data, labels
        else: data


def get_loader(batch_size):
    num_workers = 2

    labeled_data_train = DiabetesDataset('../diabetes_data/', 'labeled_train_dataset.csv', split='train', labeled=True)
    unlabeled_data_train = DiabetesDataset('../diabetes_data/', 'unlabeled_train_dataset.csv', split='train', labeled=False)
    diabetes_test = DiabetesDataset('../diabetes_data/', 'normalised_test_dataset.csv', split='test', labeled=False)

    labeled_loader_train = DataLoader(
        dataset=labeled_data_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    unlabeled_loader_train = DataLoader(
        dataset=unlabeled_data_train,
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

    return labeled_loader_train, unlabeled_loader_train, diabetes_loader_test

labeled_loader_train, unlabeled_loader_train, diabetes_loader_test = get_loader(batch_size=batch_size)
labeled_patient_iter = iter(labeled_loader_train)
unlabeled_patient_iter = iter(unlabeled_loader_train)
labeled_patient, _ = labeled_patient_iter.next()
unlabeled_patient = unlabeled_patient_iter.next()

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

def _one_hot(x):
        label_numpy = x.data.cpu().numpy()
        label_onehot = np.zeros((label_numpy.shape[0], num_classes))
        label_onehot[np.arange(label_numpy.shape[0]), label_numpy] = 1
        label_onehot = _to_var(torch.FloatTensor(label_onehot))
        return label_onehot

def get_label_mask(labeled_rate, batch_size):
    label_mask = np.zeros([batch_size], dtype=np.float32)
    label_count = np.int(batch_size * labeled_rate)
    label_mask[range(label_count)] = 1.0
    np.random.shuffle(label_mask)
    return label_mask

class Generator(nn.Module):

    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is Z, going into convolution
            nn.ConvTranspose2d(nz, ngf * 8, 2, (1, 2), 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.2),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 1, (1, 1), 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 4),
            nn.Dropout(0.2),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 2, (1, 3), 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 2),
            nn.Dropout(0.2),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 2, ngf, 1, (1, 2), 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf),
            nn.Dropout(0.2),
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
            nn.utils.weight_norm(nn.Conv2d(ndf*2, ndf*4, 1, padding=0, bias=False)),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv2d(ndf*4, ndf*4, 1, padding=0, bias=False)),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv2d(ndf*4, ndf*4, 1, padding=0, bias=False)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

        )

        self.features = nn.AvgPool2d(kernel_size=2)

        self.class_logits = nn.Linear(
            in_features=(ndf * 4) * 1 * 1,
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


optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta, 0.999), weight_decay=1)
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta, 0.999))

schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=[500, 1000, 1500, 2000, 2500], gamma=0.1)
schedulerG = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=[500, 1000, 1500, 2000, 2500], gamma=0.1)

best_disc_loss = 1
best_gen_loss = 1
best_accuracy = 0
best_epoch_number = 0

for epoch in range(num_epochs):


    schedulerD.step()
    schedulerG.step()

    masked_correct = 0
    num_samples = 0
    # For each batch in the dataloader

    for i, data in enumerate(labeled_loader_train):
        labeled_data, labels = data
        labeled_data = _to_var(labeled_data).float()
        labels = torch.FloatTensor(labels)
        labels = _to_var(labels).long().squeeze()

        logits_lab, _ = netD(labeled_data)
        loss_lab = torch.mean(d_gan_criterion(logits_lab, labels))


    for i, data in enumerate(diabetes_unlabeled_train):

        unlabeled_data = _to_var(unlabeled_data).float()


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

        mask = get_label_mask(labeled_rate, batch_size)
        mask = _to_var(torch.FloatTensor(mask))

        ##########################
        # FIRST SORT OUT SUPERVISED LOSS:
        # This is checking how well the discriminator can categorise the real data
        ##########################


        netD.zero_grad()
        logits_unl, layer_real = netD(unlabeled_data)

        logits_gen, _ = netD(generator_input.detach())
        logits_gen_adv, _ = netD(gen_adv.detach())

        l_unl = torch.logsumexp(logits_unl, 1)
        l_gen = torch.logsumexp(logits_gen, 1)


        loss_unl = - 0.5 * torch.mean(l_unl) \
                         + 0.5 * torch.mean(F.softplus(l_unl)) \
                         + 0.5 * torch.mean(F.softplus(l_gen))

        #epsilon = 1e-8

        #prob_real_be_real = 1 - real_real_1[:, -1] + epsilon
        #tmp_log = torch.log(prob_real_be_real)
        #unsupervised_loss_1 = -1 * torch.mean(tmp_log)

        #prob_fake_be_fake = fake_fake[:, -1] + epsilon
        #tmp_log = torch.log(prob_fake_be_fake)
        #unsupervised_loss_2 = -1 * torch.mean(tmp_log)

        #total_unsupervised_loss = unsupervised_loss_1 + unsupervised_loss_2

        manifold_diff = logits_gen - logits_gen_adv

        manifold = torch.sum(torch.sqrt((manifold_diff ** 2) + 1e-8))

        j_loss = torch.mean(manifold)

        loss_d = loss_unl + loss_lab + (0.001 * j_loss)


        loss_d.backward(retain_graph=True)
        optimizerD.step()

        ##########################
        # Now train the Generator
        # This is based on the feature differences between real and fake
        ##########################

        netG.zero_grad()

        _, layer_fake = netD(generator_input)


        m1 = torch.mean(layer_real, dim=0).squeeze()
        m2 = torch.mean(layer_fake, dim=0).squeeze()


        loss_g = torch.mean(torch.abs(m1 - m2))


        #prob_fake_be_real = 1 - fake_real[:, -1] + epsilon
        #tmp_log = torch.log(prob_fake_be_real)
        #loss_g_2 = -1 * torch.mean(tmp_log)

        #loss_g = loss_g_1 + loss_g_2

        loss_g.backward()
        optimizerG.step()


        pred_class = torch.argmax(logits_lab, 1)
        print(pred_class)
        correct_pred = torch.eq(pred_class, labels)
        train_accuracy = torch.mean(correct_pred.float())

        if i % 50 == 0:
            print('Training:\tepoch {}/{}\tdiscr. gan loss {}\tdiscr. class loss {}\tgen loss {}\tsamples {}/{}'.
                  format(epoch, num_epochs,
                         loss_unl.item(), loss_lab.item(),
                         loss_g.item(), i + 1,
                         len(diabetes_loader_train)))
            real_cpu, _, _ = data
            #vutils.save_image(real_cpu,
            #        './.gitignore/output/SS_GAN_TEST/real_samples.png',
            #        normalize=True)
            fake = netG(fixed_noise)
            #vutils.save_image(fake.detach(),
            #        './.gitignore/output/SS_GAN_TEST/fake_samples_epoch_%03d.png' % epoch,
            #        normalize=True)

            print('Training:\tepoch {}/{}\taccuracy {}'.format(epoch, num_epochs, train_accuracy))


    #with torch.no_grad():
    #    for i, data in enumerate(diabetes_loader_test):
    #        test_values, test_labels = data
    #        test_values = _to_var(test_values).float()
    #        test_labels = _to_var(test_labels).float()
    #        test_logits, _ = netD(test_values)
    #        pred_class = torch.argmax(test_logits, 1)
    #        correct_pred = torch.eq(pred_class.float(), test_labels)
    #        test_accuracy = torch.mean(correct_pred.float())

    #        print(f'Testing:\tepoch {epoch}/{num_epochs}\taccuracy {test_accuracy}')
    test_dataset = pd.read_csv("../diabetes_data/normalised_test_dataset.csv").values

    np.random.shuffle(test_dataset)

    test_labels = test_dataset[:, 6]
    test_dataset = test_dataset[:, 1:6]


    test_dataset = np.expand_dims(test_dataset, 1)
    test_dataset = np.expand_dims(test_dataset, 1)

    print(test_dataset.shape)

    netD.train(False)
    test_dataset = torch.tensor(test_dataset).float()
    test_labels = torch.tensor(test_labels).float()
    test_logits, _ = netD(test_dataset)
    pred_class = torch.argmax(test_logits, 1)
    correct_pred = torch.eq(pred_class.float(), test_labels)
    test_accuracy = torch.mean(correct_pred.float())
    print(f"Testing accuracy for epoch# {epoch} :{test_accuracy}")

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_epoch_number = epoch
        disc_state = {
            'epoch': epoch,
            'state_dict_disc': netD.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'loss': loss_d,
            'accuracy': test_accuracy
        }
        torch.save(disc_state, "best_disc_model.pkl")
    if loss_g < best_gen_loss:
        best_gen_loss = loss_g
        best_epoch_number = epoch
        gen_state = {
            'epoch': epoch,
            'state_dict_gen': netG.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'loss': loss_g
        }
        torch.save(gen_state, "best_gen_model.pkl")
