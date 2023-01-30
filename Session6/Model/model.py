import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    '''
    The class responsible for defining the blocks in the neural network and the forward pass of it
    '''
    def __init__(self, dropout):
        '''

        :param dropout: dropout rate
        '''
        super(Net, self).__init__()

        #------------------------------------Convolution Block 1--------------------------------------------------
        # Input Dimension : 32 * 32 * 3 (32 * 32 color images)
        # Initial Receptive Field: 1
        # Initial Jump-In (jin) : 1
        self.conv1 = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout),

            # Input Dimension : 32 * 32 * 3
            # Output Dimension: (32 + (2 * 1) - 3)/1 + 1 -> 32 * 32 * 32
            # RF Calculation: 1 + (3 - 1) * 1 -> 3 * 3
            # JOut: 1 * 1 = 1

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout),

            # Input Dimension : 32 * 32 * 32
            # Output Dimension: (32 + (2 * 1) - 3)/1 + 1 -> 32 * 32 * 64
            # RF Calculation: 3 + (3 - 1) * 1 -> 5 * 5
            # JOut: 1 * 1 = 1

            nn.Conv2d(64, 32, 1, stride=2),
            nn.ReLU()

            # Input Dimension : 32 * 32 * 64
            # Output Dimension: (32 + (2 * 0) - 1)/2 + 1 -> 16 * 16 * 32
            # RF Calculation: 5 + (1 - 1) * 1 -> 5 * 5
            # JOut: 1 * 2 = 2
            # As per requirement, we have used convolution with stride as 2

        )
        # ------------------------------------------------------------------------------------------------------

        #------------------------------------Convolution Block 2--------------------------------------------------
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout),

            # Input Dimension : 16 * 16 * 32
            # Output Dimension: (16 + (2 * 1) - 3)/1 + 1 -> 16 * 16 * 32
            # RF Calculation: 5 + (3 - 1) * 2 -> 9 * 9
            # JOut: 2 * 1 = 2


            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),

            # Input Dimension : 16 * 16 * 32
            # Output Dimension: (16 + (2 * 1) - 3)/1 + 1 -> 16 * 16 * 32
            # RF Calculation: 9 + (3 - 1) * 2 -> 13 * 13
            # JOut: 2 * 1 = 2
            # We have used depthwise separable convolution

            nn.Conv2d(32, 64, 1, padding=1, bias=False),

            # Input Dimension : 16 * 16 * 32
            # Output Dimension: (16 + (2 * 1) - 1)/1 + 1 -> 18 * 18 * 64
            # RF Calculation: 13 + (1 - 1) * 2 -> 13 * 13
            # JOut: 2 * 1 = 2

            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout),

            nn.Conv2d(64, 32, 1, stride=2),  # Input: 18x18x32 | Output: 9x9x64 | RF: 13x13
            nn.ReLU()

            # Input Dimension : 18 * 18 * 64
            # Output Dimension: (18 + (2 * 0) - 1)/2 + 1 -> 9 * 9 * 32
            # RF Calculation: 13 + (1 - 1) * 2 -> 13 * 13
            # JOut: 2 * 2 = 4
        )
        # ------------------------------------------------------------------------------------------------------



        #------------------------------------Convolution Block 3--------------------------------------------------
        self.conv3 = nn.Sequential(

            ## Dilation Block
            nn.Conv2d(32, 64, 3, padding=1, bias=False, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout),

            # Input Dimension : 9 * 9 * 32
            # Output Dimension: (9 + (2 * 1) - (3 + 2))/1 + 1 -> 7 * 7 * 64
            # RF Calculation: 13 + ((3 + 2) - 1) * 4 -> 29 * 29
            # JOut: 4 * 2 = 8
            # We have used dilation block as per requirement

            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout),

            # Input Dimension : 7 * 7 * 64
            # Output Dimension: (7 + (2 * 1) - 3)/1 + 1 -> 7 * 7 * 64
            # RF Calculation: 29 + (3 - 1) * 8 -> 45 * 45
            # JOut: 8 * 1 = 8

            nn.Conv2d(64, 16, 1, stride=2),  # Input: 7x7x64| Output: 4x4x16 | RF: 61x61
            nn.ReLU()

            # Input Dimension : 7 * 7 * 64
            # Output Dimension: (7 + (2 * 0) - 1)/2 + 1 -> 4 * 4 * 16
            # RF Calculation: 45 + (1 - 1) * 8 -> 45 * 45
            # JOut: 8 * 2 = 16
        )
        # ------------------------------------------------------------------------------------------------------

        #------------------------------------Convolution Block 4--------------------------------------------------
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout),

            # Input Dimension : 4 * 4 * 16
            # Output Dimension: (4 + (2 * 1) - 3)/1 + 1 -> 4 * 4 * 32
            # RF Calculation: 45 + (3 - 1) * 16 -> 77 * 77
            # JOut: 16 * 1 = 16

            ## Depthwise seperable Convolution2
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),

            # Input Dimension : 4 * 4 * 16
            # Output Dimension: (4 + (2 * 1) - 3)/1 + 1 -> 4 * 4 * 32
            # RF Calculation: 77 + (3 - 1) * 16 -> 109 * 109
            # JOut: 16 * 1 = 16
            # Used Depthwise seperable Convolution as per requirement

            nn.Conv2d(32, 10, 1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout2d(dropout)

            # Input Dimension : 4 * 4 * 32
            # Output Dimension: (4 + (2 * 1) - 1)/1 + 1 -> 6 * 6 * 10
            # RF Calculation: 109 + (1 - 1) * 16 -> 109 * 109
            # JOut: 16 * 1 = 16
        )
        # ------------------------------------------------------------------------------------------------------

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        '''

        :param x: input
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
