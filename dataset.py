import torch
import os
import dataset_processing
import configparser
from PIL import Image
import torchvision.transforms as transforms


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ids, path_to_data):
        
        'Initialization'
        self.ids = ids #all ids, scene_id_psv_id
        self.path_to_data = path_to_data
        #self.psvs = psvs
        #self.mpi_1_pose = mpi_1_pose
        #self.mpi_2_pose = mpi_2_pose 
        #self.target_image = target_image 
        #self.target_pose = target_pose
        #self.scene_baselineMM = baselineMM 
        #self.scene_focalLength = focalLength
        #self.scene_sensorWidthMM = sensorWidthMM

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
        mpi_1_psvs = dataset_processing.get_psvs(self.path_to_data, scene_dir, mpi_1_coords)
        mpi_2_psvs = dataset_processing.get_psvs(self.path_to_data, scene_dir, mpi_2_coords)
        target_image_coords = dataset_processing.get_target_image_coords(mpi_1_coords, mpi_2_coords)


        # Get config data
        baselineMM, focalLength, sensorWidth = dataset_processing.load_config_from_disk(self.path_to_data, scene_dir)
        
        # Scene parameters
        sample['sample_id'] = id
        sample['scene_id'] = scene_dir
        sample['baselineMM'] = baselineMM
        sample['focalLength'] = focalLength
        sample['sensorWidthMM'] = sensorWidth

        # PSVs, target_image and poses
        sample['psvs'] = torch.stack((mpi_1_psvs, mpi_2_psvs), dim=0)
        sample['psv_center_1_pose'] = dataset_processing.string2coords(mpi_1_coords)
        sample['psv_center_2_pose'] = dataset_processing.string2coords(mpi_2_coords)
        sample['target_image'] = dataset_processing.get_target_image_from_disk(self.path_to_data, scene_dir, target_image_coords)
        sample['target_image_pose'] = target_image_coords

        return sample