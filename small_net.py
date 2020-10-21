import torch 
import torch.nn as nn
import torch.nn.functional as F


class MPIPredictionNet(nn.Module):

    def __init__(self):
        super(MPIPredictionNet, self).__init__()

        self.conv1_1 = nn.Conv3d(in_channels=15, out_channels=8, kernel_size=3, stride=1, dilation=1)
        self.conv1_2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=2, dilation=1)
        self.conv2_2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=2, dilation=1)
        self.conv3_3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, dilation=1)
        self.conv4_1 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=2)

        # The input data is assumed to be of the form minibatch x channels x [optional depth] x [optional height] x width. Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor
        # Change order of dimensions necessary?
        self.nnup5 = nn.Upsample(mode='nearest', size='256')

        self.conv5_1 = nn.Conv3d(in_channels=256, out_channels=64, kernel_size=3, stride=1, dilation=1)
        self.conv7_1 = nn.Conv3d(in_channels=64, out_channels=8, kernel_size=3, stride=1, dilation=1)
        self.conv7_3 = nn.Conv3d(in_channels=8, out_channels=5, kernel_size=3, stride=1, dilation=1)

        relu = nn.ReLU()
        simoid = nn.Sigmoid()
        softmax = nn.Softmax()

    def forward(self, psv):
        
        # convolutions
        c1_1 = self.conv1_1(psv) #8
        c1_2 = self.conv1_2(c1_1) #16
        c2_2 = self.conv2_2(c1_2) #32
        c3_3 = self.conv3_3(c2_2) #64
        c4_1 = self.conv4_1(c3_3) #64

        # nearest neighbour upsamling
        nnup5 = self.nnup5(torch.cat((c4_1, c3_3), dim=1))

        # convolutions
        c5_1 = self.conv5_1(nnup5)
        c7_1 = self.conv7_1(c5_1)
        out = self.conv7_3(c7_1)

        # Split parameters
        print(out.shape)
        params = torch.split(out, split_size_or_sections=1, dim=1)

        # Output tensor is of dimension batch_size x parameters(5) x height x width x depth; 5 parameters: r,g,b,a,all-0
        alpha = self.sigmoid(params[0])
        r = self.softmax(params[1])
        g = self.softmax(params[2])
        b = self.softmax(params[3])

        return alpha, r, g, b