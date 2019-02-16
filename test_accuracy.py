import pandas as pd
import numpy as np
import torch
import torch.nn as nn

nc = 1
ndf = 64
num_classes = 2

state = torch.load("best_model.pkl")


def _to_var(x):
    x = x.cuda()
    return Variable(x)

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



classifier = Discriminator(1)

test_dataset = pd.read_csv("../diabetes_data/normalised_diabetes_dataset.csv").to_numpy()
test_dataset = test_dataset[:1780]

np.random.shuffle(test_dataset)

test_labels = test_dataset[:, 6]
test_dataset = test_dataset[:, 1:6]

test_dataset = _to_var(test_dataset)

classifier.load_state_dict(state['state_dict_disc'])


classifier.eval()

test_logits, _ = classifier(test_dataset)
pred_class = torch.argmax(test_logits, 1)
correct_pred = torch.eq(pred_class.float(), test_labels)
accuracy = torch.mean(correct_pred.float())
