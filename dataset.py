import torch
import os
import preprocessing
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
        sample = dict()
        
        # Select sample
        id = self.ids[index]
        params = str(id).split(sep='_') #0cC7GPRFAIvP5i_01_09 --> ~/less_data/0cC7GPRFAIvP5i/psv_Cam004.png'
        
        # Get scene, coords and psvs
        scene_dir = params[0]
        mpi_1_coords = params[1]
        mpi_2_coords = params[2]
        mpi_1_psvs = get_psvs(mpi_1_coords, scene_dir)
        mpi_2_psvs = get_psvs(mpi_2_coords, scene_dir)
        target_image_coords = get_target_image_coords(mpi_1_coords, mpi_2_coords)


        # Get config data
        baselineMM, focalLength, sensorWidth = load_config_from_file(scene_dir)
        
        # Scene parameters
        sample['scene_id'] = scene_dir
        sample['baselineMM'] = baselineMM
        sample['focalLength'] = focalLength
        sample['sensorWidthMM'] = sensorWidth

        # PSVs, target_image and poses
        sample['psvs'] = torch.squeeze(torch.stack((mpi_1_psvs, mpi_2_psvs), dim=0)
        sample['psv_center_1_pose'] = mpi_1_coords
        sample['psv_center_2_pose'] = mpi_2_coords
        sample['target_image'] = get_target_image_from_disk(target_image_coords)
        sample['target_image_pose'] = target_image_coords

        return sample
    
    # Function to get 5 PSVS for a center MPI position 
    # Input: MPI center position 
    # Output: Pytorch tensor with 4 nearest neighbour PSVs and the center 

    def get_psvs(mpi_coords, scene_dir):

        coords = preprocessing.string2coords(mpi_coords)
        row = coords[0]
        column coords[1]
        
        # get psvs of first mpi_pose (upperleft, ...)
        psv_1_center_coords = (row, column)
        psv_1_ul_coords = (row-1, column-1)
        psv_1_ur_coords = (row-1, column+1)
        psv_1_ll_coords = (row+1, column-1)
        psv_1_lr_coords = (row+1, column+1)

        psvs_1_center = load_psvs_from_file(psv_1_center_coords, scene_dir)
        psvs_1_ul = load_psvs_from_file(psv_1_ul_coords, scene_dir)
        psvs_1_ur = load_psvs_from_file(psv_1_ur_coords, scene_dir)
        psvs_1_ll = load_psvs_from_file(psv_1_ll_coords, scene_dir)
        psvs_1_lr = load_psvs_from_file(psv_1_lr_coords, scene_dir)

        psvs = torch.squeeze(torch.stack((psvs_1_center, psvs_1_ul, psvs_1_ur, psv_1_ll_coords, psvs_1_lr), dim=0))

        return psvs

    def load_psvs_from_file(psv_coords, scene_dir):
        path = self.path_to_data + '/' + scene_dir + '/' + 'psv_' + preprocessing.coords2string(psv_coords) + '.pt'
        psv = torch.load(path)

        # TODO: assert psv to be psv/tensor

        return psv
    
    def load_config_from_file(scene_dir):
        path = self.path_to_data + '/' + scene_dir + '/parameters.cfg'
        config = configparser.RawConfigParser()
        config.read(path)
        baselineMM = config.getfloat('extrinsics', 'baseline_mm')
        focalLength = config.getfloat('intrinsics', 'focal_length_mm')
        sensorWidth = config.getfloat('intrinsics', 'sensor_size_mm')

        return baselineMM, focalLength, sensorWidth

    def get_target_image_coords(mpi_1_coords, mpi_2_coords):

        target_coords = ''

        mpi_1_coords = preprocessing.string2coords(mpi_1_coords)
        mpi_1_row = mpi_1_coords[0]
        mpi_1_column = mpi_1_coords[1]
        mpi_2_coords = preprocessing.string2coords(mpi_2_coords)
        mpi_2_row = mpi_2_coords[0]
        mpi_2_column = mpi_2_coords[1]

        if(mpi_1_row != mpi_2_row):
            diff = abs(mpi_1_row - mpi_2_row)/2
            if(mpi_1_row > mpi_2_row):
                target_coords = (mpi_1_row-diff,mpi_1_column)
            elif(mpi_1_row < mpi_2_row):
                target_coords = (mpi_2_row-diff,mpi_2_column)
        elif(mpi_1_column != mpi_2_column):
            diff = abs(mpi_1_column - mpi_2_column)/2
            if(mpi_1_column > mpi_2_column): 
                target_coords = (mpi_1_row, mpi_1_column-diff)
            elif(mpi_1_column < mpi_2_column):
                target_coords = (mpi_2_row, mpi_2_column-diff)
        
        return target_coords
    
    def get_target_image_from_disk(target_image_coords):
        
        path = self.path_to_data + '/' + scene_dir + '/' + 'input_Cam0' + preprocessing.coords2string(target_image_coords) +'.png' 
        pil_image = Image.open(path)
        pil2tensor = transforms.ToTensor()
        rgb_image = pil2tensor(pil_image)

        return rgb_image