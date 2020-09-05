import torch
import torch.nn as nn
import torch.nn.functional as F

from mpi_prediction_net import MPIPredictionNet
from mpi_prediction_shortcut import MPIPredictionShortcut



'''
Parameter configuration

1 set = 5 images (HxW) --> 5 PSVs (HxWxD) --> 1 MPI (ParamsxHxWxD) 
in total 2 sets so batch_size = 2
'''
# image / PSV size 
height = 28
width = 28
depth = 8
# PSV stacked together along channel dimension (according to paper should be 15)
psv_1 = torch.rand((3, height, width, depth))
psv_2 = torch.rand((3, height, width, depth))
psv_3 = torch.rand((3, height, width, depth))
psv_4 = torch.rand((3, height, width, depth))
psv_5 = torch.rand((3, height, width, depth))

psvs = torch.cat((psv_1, psv_2, psv_3, psv_4, psv_5), dim=0) #(psv_1, psv_2, psv_3, psv_4, psv_5)

# Get MPI
simple_mpi = MPIPredictionShortcut.calulatesimpleMPI(psvs)

print(psvs.shape)
print(simple_mpi.shape)











