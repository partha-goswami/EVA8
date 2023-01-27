import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Net(nn.Module):
    '''
    Our neural network class, with specified layers, input, output, filters and chosen regularizations
    '''

    def __init__(self, norm_type='BN', dropout_value=0.01):
        '''
        Init method for our neural network class.
        Here we define the layers and elements.

        :param norm_type: normalization type, can be barch normalization (BN),
        group normalization (GN) or layer normalization (LN)
        :param dropout_value: dropout rate
        '''
        super(Net, self).__init__()
        self.conv1 = self.conv2d(1, 8, 3, norm_type, dropout_value, 2)
        self.conv2 = self.conv2d(8, 16, 3, norm_type, dropout_value, 4)

        self.trans1 = nn.Sequential(

            nn.MaxPool2d(2, 2),  # Input 24x24 output 12x12 RF : 6x6
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False)
        )

        self.conv3 = self.conv2d(8, 16, 3, norm_type, dropout_value, 4)
        self.conv4 = self.conv2d(16, 16, 3, norm_type, dropout_value, 4)
        self.conv5 = self.conv2d(16, 16, 3, norm_type, dropout_value, 4)
        self.avgpool2d = nn.AvgPool2d(kernel_size=6)
        self.conv6 = self.conv2d(16, 16, 1, norm_type, dropout_value, 4)
        self.conv7 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)

    def conv2d(self, in_channels, out_channels, kernel_size, norm_type, dropout, num_of_groups):
        '''
        This method creates ordered custom containers for convolutional blocks.
        Mostly, we follow convolution, relu, normalization and dropout ordering

        :param in_channels: Input channels
        :param out_channels: Output channels
        :param kernel_size: Kernel size
        :param norm_type: normalization type, BN, LN, or GN
        :param dropout: dropout rate
        :param num_of_groups: number of groups, mostly applicable for group normalization (GN), although
        we use this for layer normalization (LN) too with group value as one.
        :return: Ordered convolution blocks
        '''
        if norm_type == "BN":
            conv = nn.Sequential(OrderedDict([
                ('conv2d',
                 nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0,
                           bias=False)),
                ('Relu', nn.ReLU()),
                ('BatchNorm', nn.BatchNorm2d(out_channels)),
                ('Dropout', nn.Dropout(dropout))
            ]))
        elif norm_type == "LN":
            conv = nn.Sequential(OrderedDict([
                ('conv2d',
                 nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0,
                           bias=False)),
                ('Relu', nn.ReLU()),
                ('LayerNorm', nn.GroupNorm(1, out_channels)),
                ('Dropout', nn.Dropout(dropout))
            ]))
        elif norm_type == "GN":
            conv = nn.Sequential(OrderedDict([
                ('conv2d',
                 nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0,
                           bias=False)),
                ('Relu', nn.ReLU()),
                ('GroupNorm', nn.GroupNorm(num_of_groups, out_channels)),
                ('Dropout', nn.Dropout(dropout))
            ]))
        else:
            conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0,
                          bias=False),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        return conv

    def forward(self, x):
        '''
        This method signifies forward pass through the neural network layers.
        :param x: Input data
        :return: Log softmax output
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.trans1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool2d(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)