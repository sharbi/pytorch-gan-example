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
from PIL import Image
from sklearn.metrics import f1_score

import pickle as pkl

manual_seed = 123

# Set random seeds
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# Set initial paramaters
workers = 2
batch_size = 32
image_size = 256
num_classes = 15
nc = 1
nz = 100
ngf = 64
ndf = 64
num_epochs = 5000
lr = 0.0003
beta = 0.5
ngpu = 1
labeled_rate = 0.1

list_of_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Pleural_Thickening", "Fibrosis", "Emphysema", "Hernia", "No Finding"]

# Create dataset

class CXRDataset(Dataset):
    def __init__(self, root_dir, data_file, split, transform):
        self.split = split
        self.root_dir = root_dir
        self.transform = transform
        self.use_gpu = True if torch.cuda.is_available() else False
        self.info = pd.read_csv(root_dir + data_file)
        self.encoded_labels = list(map(self._separate_labels, self.info.iloc[:, 2]))


    def _generate_one_hot(self, label):
        output = np.zeros(15, dtype=int)
        output = np.concatenate((output, np.array([1])))
        for i, x in enumerate(list_of_labels):
            if x in label:
                output[i] = 1
        return list(output)

    def _separate_labels(self, labels):
        new_labels = []
        if "|" in labels:
            labels = labels.split("|")
        else:
            labels = [labels]
        new_labels = (self._generate_one_hot(labels))
        return new_labels


    def _is_train_dataset(self):
        return True if self.split == 'train' else False


    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.info.iloc[idx, 1])
        image = Image.open(img_name)
        labels = self.encoded_labels[idx]

        age = self.info.iloc[idx, 5]
        gender = self.info.iloc[idx, 6]
        view_position = self.info.iloc[idx, 7]

        image = self.transform(image)

        return image, torch.LongTensor(labels)

def apply_threshold(predictions):
    output = []
    for prediction in predictions:
        if prediction > 0.5:
            output.append(1)
        else:
            output.append(0)
    return output


def get_loader(batch_size):
    num_workers = 2

    normalise = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        normalise
    ])


    data_train = CXRDataset('../NIH_Images/', 'train_dataset.csv', split='train', transform=transform)
    data_test = CXRDataset('../NIH_Images/', 'test_dataset.csv', split='test', transform=transform)


    loader_train = DataLoader(
        dataset=data_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    loader_test = DataLoader(
        dataset=data_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return loader_train, loader_test

loader_train, loader_test = get_loader(batch_size=batch_size)
image_iter = iter(loader_train)
image, _ = image_iter.next()


test_iter = iter(loader_test)
test_image, _ = test_iter.next()



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
            nn.ConvTranspose2d(nz, ngf * 8, 4, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.2),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 4),
            nn.Dropout(0.2),

            nn.ConvTranspose2d(ngf * 4, ngf * 4, 4, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 4),
            nn.Dropout(0.2),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 3, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf * 2),
            nn.Dropout(0.2),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 3, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ngf),
            nn.Dropout(0.2),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf, nc, 4, 2, 0, bias=False),

            nn.Tanh()
        # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        out = self.main(input)
        return out




class Discriminator(nn.Module):

    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.conv1 = nn.utils.weight_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        self.conv2 = nn.utils.weight_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False))
        self.conv3 = nn.utils.weight_norm(nn.Conv2d(ndf*2, ndf*2, 4, 2, 1, bias=False))
        self.conv4 = nn.utils.weight_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False))
        self.conv5 = nn.utils.weight_norm(nn.Conv2d(ndf*4, ndf*4, 3, 1, 1, bias=False))
        self.conv6 = nn.utils.weight_norm(nn.Conv2d(ndf*4, ndf*4, 3, 1, 0, bias=False))

        self.features = nn.AvgPool2d(kernel_size=2)

        self.class_logits = nn.Linear(
            in_features=(ndf * 4) * 1 * 1,
            out_features=num_classes + 1)

        #self.gan_logits = _ganLogits()

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):

        layer1 = F.dropout(F.leaky_relu(self.conv1(inputs), 0.2), 0.5)
        layer2 = F.dropout(F.leaky_relu(self.conv2(layer1), 0.2), 0.5)
        layer3 = F.dropout(F.leaky_relu(self.conv3(layer2), 0.2), 0.5)
        layer4 = F.dropout(F.leaky_relu(self.conv3(layer3), 0.2), 0.5)
        layer5 = F.dropout(F.leaky_relu(self.conv3(layer4), 0.2), 0.5)
        layer6 = F.dropout(F.leaky_relu(self.conv4(layer5), 0.2), 0.5)
        layer7 = F.dropout(F.leaky_relu(self.conv5(layer6), 0.2), 0.5)
        layer8 = F.leaky_relu(self.conv6(layer7), 0.2)


        features = self.features(layer8)
        features = features.squeeze()

        class_logits = self.class_logits(features)

        #gan_logits = self.gan_logits(class_logits)

        out = self.sigmoid(class_logits)

        return class_logits, features, out


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
print(netG)

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
print(netD)

d_unsupervised_criterion = nn.BCEWithLogitsLoss()
d_gan_criterion = nn.BCEWithLogitsLoss()
fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)
fixed_noise = _to_var(fixed_noise)


d_gan_labels_real = torch.FloatTensor(batch_size)
d_gan_labels_fake = torch.FloatTensor(batch_size)

real_label = 1
fake_label = 0


optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta, 0.999), weight_decay=1)
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta, 0.999))

#schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=[500, 1000, 1500, 2000, 2500], gamma=0.1)
#schedulerG = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=[500, 1000, 1500, 2000, 2500], gamma=0.1)

best_disc_loss = 1
best_gen_loss = 1
best_accuracy = 0
best_epoch_number = 0

for epoch in range(num_epochs):


    #schedulerD.step()
    #schedulerG.step()

    masked_correct = 0
    num_samples = 0
    # For each batch in the dataloader

    for i, data in enumerate(loader_train):
        labeled_data, labels = data
        labeled_data = _to_var(labeled_data).float()

        labels = torch.LongTensor(labels)
        labels = _to_var(labels).float()

        mask = get_label_mask(labeled_rate, batch_size)
        mask = _to_var(torch.FloatTensor(mask))
        epsilon = 1e-8


        netD.zero_grad()
        logits_lab, layer_real, real_real = netD(labeled_data)

        #loss_lab = d_gan_criterion(logits_lab, labels)

        loss_lab = -torch.sum(labels * torch.sigmoid(logits_lab), dim=1)
        loss_lab = (loss_lab * mask) / torch.max(_to_var(torch.Tensor([(1.0), torch.sum(mask)])))
        loss_lab = torch.mean(loss_lab)


        noise = torch.FloatTensor(batch_size, nz, 1, 1)

        noise.resize_(batch_size, nz, 1, 1).uniform_(0, 100)
        noise_var = _to_var(noise)
        generator_input = netG(noise_var)

        #pert_input = noise.resize_(labels.data.shape[0], nz, 1, 1).normal_(0, 100)
        #pert_n = F.normalize(pert_input)

        #noise_pert = noise + 1. * pert_n
        #noise_pert = _to_var(noise_pert)
        #gen_inp_pert = netG(noise_pert)

        #manifold_regularisation_value = (gen_inp_pert - generator_input)
        #manifold_regularisation_norm = F.normalize(manifold_regularisation_value)

        #gen_adv = generator_input + 20. * manifold_regularisation_norm


        ##########################
        # FIRST SORT OUT SUPERVISED LOSS:
        # This is checking how well the discriminator can categorise the real data
        ##########################


        #logits_unl, layer_real = netD(unlabeled_data)

        logits_gen, _, fake_fake = netD(generator_input.detach())
        #logits_gen_adv, _, _ = netD(gen_adv.detach())

        l_unl = torch.logsumexp(logits_lab, 0)
        l_gen = torch.logsumexp(logits_gen, 0)


        loss_unl = - 0.5 * torch.mean(l_unl) \
                         + 0.5 * torch.mean(F.softplus(l_unl)) \
                         + 0.5 * torch.mean(F.softplus(l_gen))


        #manifold_diff = logits_gen - logits_gen_adv

        #manifold = torch.sum(torch.sqrt((manifold_diff ** 2) + 1e-8))

        #j_loss = torch.mean(manifold)

        loss_d = loss_unl + loss_lab


        loss_d.backward(retain_graph=True)
        optimizerD.step()

        ##########################
        # Now train the Generator
        # This is based on the feature differences between real and fake
        ##########################

        netG.zero_grad()

        _, layer_fake, fake_real = netD(generator_input)


        m1 = torch.mean(layer_real, dim=0).squeeze()
        m2 = torch.mean(layer_fake, dim=0).squeeze()

        feature_distance = m1 - m2

        loss_g_1 = torch.mean(torch.matmul(feature_distance, feature_distance))

        fake_reals = torch.FloatTensor([batch_size, 1]).uniform_(0.9, 1.2)
        fake_reals = _to_var(fake_reals).float()

        prob_fake_be_real = 1 - fake_real[:, -1] + epsilon
        tmp_log = torch.log(prob_fake_be_real)
        loss_g_2 = -1 * torch.mean(tmp_log)


        loss_g = loss_g_1 + loss_g_2

        loss_g.backward()
        optimizerG.step()


        thresholder_predictions = torch.sigmoid(logits_lab)
        preds = map(apply_threshold, thresholder_predictions)
        #f1 = f1_score(labels, list(preds))
        #print(f1)
        total = len(labels) * num_classes
        correct = (list(preds) == labels.cpu().numpy().astype(int)).sum()
        train_accuracy = 100 * correct / total

        if i % 50 == 0:
            print('Training:\tepoch {}/{}\tdiscr. gan loss {}\tdiscr. class loss {}\tgen loss {}\tsamples {}/{}'.
                  format(epoch, num_epochs,
                         loss_unl.item(), loss_lab.item(),
                         loss_g.item(), i + 1,
                         len(loader_train)))
            real_cpu, _ = data
            vutils.save_image(real_cpu,
                    './.gitignore/output/SS_GAN_TEST/real_samples.png',
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                    './.gitignore/output/SS_GAN_TEST/fake_samples_epoch_%03d.png' % epoch,
                    normalize=True)

            print('Training:\tepoch {}/{}\taccuracy {}'.format(epoch, num_epochs, train_accuracy))


    with torch.no_grad():
        for i, data in enumerate(loader_test):
            test_values, test_labels = data
            test_values = _to_var(test_values).float()
            test_labels = _to_var(test_labels).float()
            test_logits, _, _ = netD(test_values)
            test_thresholder_predictions = torch.sigmoid(test_logits)
            test_preds = map(apply_threshold, test_thresholder_predictions)
            total = len(labels) * num_classes
            correct = (list(test_preds) == test_labels.cpu().numpy().astype(int)).sum()
            test_accuracy = 100 * correct / total

            print(f'Testing:\tepoch {epoch}/{num_epochs}\taccuracy {test_accuracy}')

    if train_accuracy > best_accuracy:
        best_accuracy = train_accuracy
        best_epoch_number = epoch
        disc_state = {
            'epoch': epoch,
            'state_dict_disc': netD.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'loss': loss_d,
            'accuracy': train_accuracy
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
