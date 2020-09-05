import torch 
import torch.nn as nn
import torch.nn.functional as F

class MPIPredictionShortcut:

    def calulatesimpleMPI(psvs):   
        print('-------------------------------------------------------------')
        print('MPIPredictionShortcut - calculate simple MPI')
        
        # Create MPI with same dimensions as PSV
        height = psvs.shape[1]
        width = psvs.shape[2]
        depth = psvs.shape[3]
        channels = 3 

        simple_mpi = torch.zeros((channels, height, width, depth))

        # Calculate for every voxel of MPI the arith. mean of PSVs at the same position 
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    for d in range(depth):
                        simple_mpi[c, h,w,d] =  1/5 * (psvs[c,h,w,d] + psvs[(3+c),h,w,d] + psvs[(6+c),h,w,d] + psvs[(9+c),h,w,d] + psvs[(12+c),h,w,d])
        
        return simple_mpi


        

