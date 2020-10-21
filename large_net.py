import torch 
import torch.nn as nn
import torch.nn.functional as F


class MPIPredictionNet(nn.Module):

    def __init__(self):
        super(MPIPredictionNet, self).__init__()

        self.conv1_1 = nn.Conv3d(in_channels=15, out_channels=8, kernel_size=3, stride=(1,1,1), padding=(1,1,1), dilation=1)
        self.conv1_2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=(2,2,1), padding=(1,1,1),dilation=1)
        self.conv2_1 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=(1,1,1), padding=(1,1,1),dilation=1)
        self.conv2_2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=(2,2,1), padding=(1,1,1),dilation=1)
        self.conv3_1 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=(1,1,1), padding=(1,1,1),dilation=1)
        self.conv3_2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=(1,1,1), padding=(1,1,1),dilation=1)
        self.conv3_3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=(2,2,1), padding=(1,1,1),dilation=1)
        self.conv4_1 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=(1,1,1), padding=(2,2,2),dilation=2)
        self.conv4_2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=(1,1,1), padding=(2,2,2),dilation=2)
        self.conv4_3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=(1,1,1), padding=(2,2,2),dilation=2)

        self.nnup5 = nn.Upsample(mode='nearest', scale_factor=(2,2,1))

        self.conv5_1 = nn.Conv3d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=(1,1,1), dilation=1)
        self.conv5_2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=(1,1,1), dilation=1)
        self.conv5_3 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=(1,1,1), dilation=1)

        self.nnup6 = nn.Upsample(mode='nearest', scale_factor=(2,2,1))

        self.conv6_1 = nn.Conv3d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=(1,1,1), dilation=1)
        self.conv6_2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=(1,1,1), dilation=1)

        self.nnup7 = nn.Upsample(mode='nearest', scale_factor=(2,2,1))

        self.conv7_1 = nn.Conv3d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=(1,1,1), dilation=1)
        self.conv7_2 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=(1,1,1), dilation=1)
        self.conv7_3 = nn.Conv3d(in_channels=8, out_channels=5, kernel_size=3, stride=1, padding=(1,1,1), dilation=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, psv):
        c1_1 = self.relu(self.conv1_1(psv))
        c1_2 = self.relu(self.conv1_2(c1_1))
        c2_1 = self.relu(self.conv2_1(c1_2))
        c2_2 = self.relu(self.conv2_2(c2_1))
        c3_1 = self.relu(self.conv3_1(c2_2))
        c3_2 = self.relu(self.conv3_2(c3_1))
        c3_3 = self.relu(self.conv3_3(c3_2))
        c4_1 = self.relu(self.conv4_1(c3_3))
        c4_2 = self.relu(self.conv4_2(c4_1))
        c4_3 = self.relu(self.conv4_3(c4_2))

        # 2xnearest neighbour upsampling?
        concat_nnup5 = torch.cat((c4_3, c3_3), dim=1)
        nnup5 = self.relu(self.nnup5(concat_nnup5))

        c5_1 = self.relu(self.conv5_1(nnup5))
        c5_2 = self.relu(self.conv5_2(c5_1))
        c5_3 = self.relu(self.conv5_3(c5_2))

        # 2xnearest neighbour upsampling?
        concat_nnup6 = torch.cat((c5_3, c2_2), dim=1)
        nnup6 = self.relu(self.nnup6(concat_nnup6))

        c6_1 = self.relu(self.conv6_1(nnup6))
        c6_2 = self.relu(self.conv6_2(c6_1))

        # 2xnearest neighbour upsampling?
        concat_nnup7 = torch.cat((c6_2, c1_2), dim=1)
        nnup7 = self.relu(self.nnup7(concat_nnup7))

        c7_1 = self.relu(self.conv7_1(nnup7))
        c7_2 = self.relu(self.conv7_2(c7_1))

        out = self.conv7_3(c7_2)
        
        # Split parameters
        params = torch.split(out, split_size_or_sections=1, dim=1)
        
        # Output tensor is of dimension batch_size x parameters(5) x height x width x depth; 5 parameters: r,g,b,a,all-0
        alpha = self.sigmoid(params[0])
        r = self.softmax(params[1])
        g = self.softmax(params[2])
        b = self.softmax(params[3])

        # Stack together
        mpis = torch.squeeze(torch.stack((r, g, b, alpha), dim=1))

        return mpis