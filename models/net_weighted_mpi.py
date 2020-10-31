import torch 
import torch.nn as nn
import torch.nn.functional as F


class MPIPredictionNet(nn.Module):

    def __init__(self, device):
        super(MPIPredictionNet, self).__init__()

        self.conv1_1 = nn.Conv3d(in_channels=15, out_channels=8, kernel_size=3, stride=(1,1,1), padding=(1,1,1), dilation=1)
        self.bn_c1_1 = nn.BatchNorm3d(8)
        self.conv1_2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=(2,2,1), padding=(1,1,1),dilation=1)
        self.bn_c1_2 = nn.BatchNorm3d(16)
        self.conv2_2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=(2,2,1), padding=(1,1,1),dilation=1)
        self.bn_c2_2 = nn.BatchNorm3d(32)
        self.conv3_3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=(2,2,1), padding=(1,1,1),dilation=1)
        self.bn_c3_3 = nn.BatchNorm3d(64)
        self.conv4_3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=(1,1,1), padding=(2,2,2),dilation=2)
        self.bn_c4_3 = nn.BatchNorm3d(64)

        self.nnup5 = nn.Upsample(mode='nearest', scale_factor=(2,2,1))
        self.bn_nnup5 = nn.BatchNorm3d(128)

        self.conv5_1 = nn.Conv3d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=(1,1,1), dilation=1)
        self.bn_c5_1 = nn.BatchNorm3d(32)
       
        self.nnup6 = nn.Upsample(mode='nearest', scale_factor=(2,2,1))
        self.bn_nnup6 = nn.BatchNorm3d(64)

        self.conv6_1 = nn.Conv3d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=(1,1,1), dilation=1)
        self.bn_c6_1 = nn.BatchNorm3d(16)

        self.nnup7 = nn.Upsample(mode='nearest', scale_factor=(2,2,1))
        self.bn_nnup7 = nn.BatchNorm3d(32)

        #self.conv7_1 = nn.Conv3d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=(1,1,1), dilation=1)
        self.conv7_1 = nn.Conv3d(in_channels=32, out_channels=5, kernel_size=3, stride=1, padding=(1,1,1), dilation=1)
        #self.bn_c7_1 = nn.BatchNorm3d(8)

        #self.conv7_3 = nn.Conv3d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=(1,1,1), dilation=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax(dim=4)
        self.softmax = nn.Softmax()

        self.device = device


    def forward(self, psvs):

        # layern normalization
        #TODO m = nn.LayerNorm(psvs.size()[1:])
        #m = nn.LayerNorm(psvs.size()[1:])

        c1_1 = self.relu(self.bn_c1_1(self.conv1_1(psvs)))
        c1_2 = self.relu(self.bn_c1_2(self.conv1_2(c1_1)))
        c2_2 = self.relu(self.bn_c2_2(self.conv2_2(c1_2)))
        c3_3 = self.relu(self.bn_c3_3(self.conv3_3(c2_2)))
        c4_3 = self.relu(self.bn_c4_3(self.conv4_3(c3_3)))

        # 2xnearest neighbour upsampling?
        concat_nnup5 = torch.cat((c4_3, c3_3), dim=1)
        nnup5 = self.relu(self.bn_nnup5(self.nnup5(concat_nnup5)))

        c5_1 = self.relu(self.bn_c5_1(self.conv5_1(nnup5)))

        # 2xnearest neighbour upsampling?
        concat_nnup6 = torch.cat((c5_1, c2_2), dim=1)
        nnup6 = self.relu(self.bn_nnup6(self.nnup6(concat_nnup6)))

        c6_1 = self.relu(self.bn_c6_1(self.conv6_1(nnup6)))

        #print(torch.cuda.memory_summary(device=None, abbreviated=False))

        # 2xnearest neighbour upsampling?
        concat_nnup7 = torch.cat((c6_1, c1_2), dim=1)
        nnup7 = self.relu(self.bn_nnup7(self.nnup7(concat_nnup7)))

        out = self.conv7_1(nnup7)

        mpis = self.network_into_mpi(out, psvs)


        return mpis
    

    def network_into_mpi(self, tensor, psvs):
    
        #assume tensor is shape (2, 5,512,512,8)
        #assume psvs is shape (2,5,3,512,512)
        
        psvs = torch.reshape(psvs, (2,5,3,512,512,8))

        mpis = torch.zeros((2,4,512,512,8))
        
        mpis[:,0] = tensor[:,0]
        
        softmax_input1 = torch.stack( [torch.zeros((512,512,8)).to(self.device), tensor[0,1], tensor[0,2], tensor[0,3], tensor[0,4]], dim=0)
        softmax_output1 = torch.nn.functional.softmax(softmax_input1, dim=0)
        
        softmax_input2 = torch.stack( [torch.zeros((512,512,8)).to(self.device), tensor[1,1], tensor[1,2], tensor[1,3], tensor[1,4]], dim=0)
        softmax_output2 = torch.nn.functional.softmax(softmax_input2, dim=0)
        
        for d in range(8):
            mpis[0,1:,:,:,d] = torch.sum(psvs[0,:,:,:,:] * softmax_output1[:,None,:,:,d], dim=0)
            mpis[1,1:,:,:,d] = torch.sum(psvs[1,:,:,:,:] * softmax_output2[:,None,:,:,d], dim=0)
                        
        return mpis
    
    