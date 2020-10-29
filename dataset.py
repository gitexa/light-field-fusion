import torch
import os
import dataset_processing
import configparser
from PIL import Image
import torchvision.transforms as transforms

'''
Pytorch Dataset Class
- one datapoint consists of 2 x 5 PSVs 
'''
class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ids, path_to_data, layers):
        
        'Initialization'
        self.ids = ids #all ids, scene_id_psv_id
        self.path_to_data = path_to_data
        self.layers = layers

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ids)

    def __getitem__(self, index):
        
        # Generate one sample of data: 2x5 PSVS with hold-out target-view 
        sample = {}
        
        # Select sample
        id = self.ids[index]
        params = str(id).split(sep='_') #0cC7GPRFAIvP5i_01_09 --> ~/less_data/0cC7GPRFAIvP5i/psv_Cam004.png'
        
        # Get scene, coords and psvs
        scene_dir = params[0]
        mpi_1_coords = params[1]
        mpi_2_coords = params[2]
        mpi_1_psvs, mpi_1_min_disp, mpi_1_bin_size = dataset_processing.get_psvs(self.path_to_data, scene_dir, mpi_1_coords)
        mpi_2_psvs, mpi_2_min_disp, mpi_2_bin_size = dataset_processing.get_psvs(self.path_to_data, scene_dir, mpi_2_coords)
        target_image_coords = dataset_processing.get_target_image_coords(mpi_1_coords, mpi_2_coords)

        # Get config data
        baselineMM, focalLength, sensorWidth, focus_distance_m= dataset_processing.load_config_from_disk(self.path_to_data, scene_dir)
        
        # Scene parameters
        sample['sample_id'] = id
        sample['scene_id'] = scene_dir
        sample['baselineMM'] = baselineMM
        sample['focalLength'] = focalLength
        sample['sensorWidthMM'] = sensorWidth
        sample['focus_distance_m'] = focus_distance_m
        sample['layers'] = self.layers

        #TODO binsize, mindisp
        
        # MPI 1 metadata
        sample['mpi_1_bin_size'] = mpi_1_bin_size
        sample['mpi_1_min_disp'] = mpi_1_min_disp

        # MPI 2 metadata
        sample['mpi_2_bin_size'] = mpi_2_bin_size
        sample['mpi_2_min_disp'] = mpi_2_min_disp

        # PSVs poses
        sample['psvs'] = torch.stack((mpi_1_psvs, mpi_2_psvs), dim=0)
        sample['psv_center_1_pose'] = dataset_processing.string2coords(mpi_1_coords)
        sample['psv_center_2_pose'] = dataset_processing.string2coords(mpi_2_coords)

        # Target image
        sample['target_image'] = dataset_processing.get_target_image_from_disk(self.path_to_data, scene_dir, target_image_coords)
        sample['target_image_pose'] = target_image_coords

        return sample
