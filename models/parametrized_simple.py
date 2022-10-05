import torch.nn as nn
import torch.nn.functional as F
import torch
from models.model import Model


class ParametrizedSimpleNet(Model):
    def __init__(self, num_classes, out_channels1=32, out_channels2=64,
                    kernel_size1=3, kernel_size2=3,
                    strides1=1, strides2=1,
                    dropout1=0.25, dropout2=0.5,
                    fc1=128, max_pool=2, activation='relu'):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, out_channels1, kernel_size1, strides1)
        self.conv2 = nn.Conv2d(32, out_channels2, kernel_size2, strides2)
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.max_pool = max_pool
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'selu':
            self.activation = F.selu

        out1 = 28 - strides1 * 2
        out2 = (out1 - strides2 * 2) // max_pool

        self.fc1 = nn.Linear(out_channels2 * out2 * out2, fc1)
        self.fc2 = nn.Linear(fc1, num_classes)

    def forward(self, x, latent=False):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = F.max_pool2d(x, self.max_pool)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
