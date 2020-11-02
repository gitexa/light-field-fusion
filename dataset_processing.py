import os
import torch
import configparser
from PIL import Image
import torchvision.transforms as transforms


'Helper Methods'
def get_all_scenes(path):
    
    scenes = os.listdir(path)
    all_scenes = list()
    
    # Loop to print each filename separately
    for scene in scenes:
        all_scenes.append(scene)
    
    scenes_count = len(all_scenes)
    print(f"Found {scenes_count} scenes.")

    return all_scenes



def parse_camsnumbering2coordinates(number):

    #int_num = 51
    int_num = int(number)
    column = int(int_num % 9)
    row = int((int_num - column)/9)

    coord = (row, column)

    return coord

# (row, column)
def parse_coordinates2camsnumbering(coordinates):
    
    cam_number = ''

    row = int(coordinates[0])
    column = int(coordinates[1])
    number = 9 * row + column

    return '0' + str(number)


# Function to transfrom coordinate tuples to string (1,1) --> 11
# Input: coordinates 
# Output: string 

def coords2string(coords):
    s = ''.join((str(coord) for coord in coords))
    return s

def string2coords(s):
    l = list(str(s))
    return (int(l[0]), int(l[1]))


# Generate all IDs for the given scenario with fixed geometry in a 9x9 camera grid - 2 MPI positions as center of a cross with target_view in between
# --> format of an ID is "sceneid_mpicoords1_mpicoordsposition2", e.g. "1JQ8tLwWMnJtSO_11_15"
# total count 6*6*2
# Input: all scenes
# Output: IDs of all training-data

def generate_all_ids(all_scenes):
    
    all_ids= []

    for scene in all_scenes:
        for mpi_1_row in range(1,8):
            for mpi_1_column in range(1,8):
                # First MPI coords
                mpi_1 = (mpi_1_row, mpi_1_column)
                # Get both samples 
                mpi_2_opt1, mpi_2_opt2 = get_second_mpi_position(mpi_1_row, mpi_1_column)
                # Generate two IDs and append to all_id list
                if(mpi_2_opt1 != ''):
                    id_1 = str(scene) + '_' + coords2string(mpi_1) + '_' + coords2string(mpi_2_opt1)
                    all_ids.append(id_1)
                if(mpi_2_opt2 != ''):
                    id_2 = str(scene) + '_' + coords2string(mpi_1) + '_' + coords2string(mpi_2_opt2)
                    all_ids.append(id_2)
    
    return all_ids


# Function to get 2 positions for the second MPI, given the first MPI
# Input: row and colum coordinates of first MPI
# Output: two possible coordinate-tuples (row,column) for the second MPI position
def get_second_mpi_position(row, column):

    second_mpi_pose_var1 = ''
    second_mpi_pose_var2 = ''
    
    # select second MPI pose
    if(row<=3 and column <=3):
        second_mpi_pose_var1 = (row, column+4)
        second_mpi_pose_var2 = (row+4, column)
    elif(row<=3 and column>4):
        second_mpi_pose_var1 = (row, column-4)
        second_mpi_pose_var2 = (row+4, column)  
    elif(row>4 and column<=3):
        second_mpi_pose_var1 = (row, column+4)
        second_mpi_pose_var2 = (row-4, column)
    elif(row>4 and column>4):
        second_mpi_pose_var1 = (row, column-4)
        second_mpi_pose_var2 = (row-4, column)
    # edge case row/column==4
    elif(row==4 and column<=3):
        second_mpi_pose_var1 = (row, column+4)
    elif(row==4 and column>4):
        second_mpi_pose_var1 = (row, column-4)
    elif(row<=3 and column==4):
        second_mpi_pose_var1 = (row+4, column)
    elif(row>4 and column==4):
        second_mpi_pose_var1 = (row-4, column)
    
    
    #sample_1 = (first_mpi_pose, second_mpi_pose_var1)
    #sample_2 = (first_mpi_pose, second_mpi_pose_var2)
    #nn_sample_1 = get_nn(sample_1)
    #nn_sample_2 = get_nn(sample_2)

    return second_mpi_pose_var1, second_mpi_pose_var2

    
def get_target_image_coords(mpi_1_coords, mpi_2_coords):

    target_coords = ''

    mpi_1_coords = string2coords(mpi_1_coords)
    mpi_1_row = mpi_1_coords[0]
    mpi_1_column = mpi_1_coords[1]
    mpi_2_coords = string2coords(mpi_2_coords)
    mpi_2_row = mpi_2_coords[0]
    mpi_2_column = mpi_2_coords[1]

    if(mpi_1_row != mpi_2_row):
        diff = int(abs(mpi_1_row - mpi_2_row)/2)
        if(mpi_1_row > mpi_2_row):
            target_coords = (mpi_1_row-diff,mpi_1_column)
        elif(mpi_1_row < mpi_2_row):
            target_coords = (mpi_2_row-diff,mpi_2_column)
    elif(mpi_1_column != mpi_2_column):
        diff = int(abs(mpi_1_column - mpi_2_column)/2)
        if(mpi_1_column > mpi_2_column): 
            target_coords = (mpi_1_row, mpi_1_column-diff)
        elif(mpi_1_column < mpi_2_column):
            target_coords = (mpi_2_row, mpi_2_column-diff)
        
    return target_coords
    
# Function to get 5 PSVS for a center MPI position 
# Input: MPI center position 
# Output: Pytorch tensor with 4 nearest neighbour PSVs and the center 

def get_psvs(path_to_data, scene_dir, mpi_coords):

    coords = string2coords(mpi_coords)
    row = coords[0]
    column = coords[1]
    
    # get psvs of first mpi_pose (upperleft, ...)
    psv_center_coords = (row, column)
    psv_ul_coords = (row-1, column-1)
    psv_ur_coords = (row-1, column+1)
    psv_ll_coords = (row+1, column-1)
    psv_lr_coords = (row+1, column+1)

    psv_center, psv_center_min_disp, psv_center_bin_size = load_psvs_from_disk(path_to_data, scene_dir, psv_center_coords)
    psv_ul, psv_ul_min_disp, psv_ul_bin_size = load_psvs_from_disk(path_to_data, scene_dir, psv_ul_coords)
    psv_ur, psv_ur_min_disp, psv_ur_bin_size = load_psvs_from_disk(path_to_data, scene_dir, psv_ur_coords)
    psv_ll, psv_ll_min_disp, psv_ll_bin_size = load_psvs_from_disk(path_to_data, scene_dir, psv_ll_coords)
    psv_lr, psv_lr_min_disp, psv_lr_bin_size = load_psvs_from_disk(path_to_data, scene_dir, psv_lr_coords)

    psvs = torch.cat((psv_center, psv_ul, psv_ur, psv_ll, psv_lr), dim=0)

    return psvs, psv_center_min_disp, psv_center_bin_size

def load_psvs_from_disk(path_to_data, scene_dir, psv_coords):
    path = path_to_data + '/' + scene_dir + '/' + 'psv_' + parse_coordinates2camsnumbering(psv_coords) + '.pt'

    #TODO: load correct PSV and assert psv to be psv/tensor
    psv_package = torch.load(path)
    psv = psv_package['psv']
    min_disp = psv_package['min_disp']
    bin_size = psv_package['bin_size']

    #psv = torch.rand(3,128,128,8)

    return psv, min_disp, bin_size

def load_config_from_disk(path_to_data, scene_dir):
    path = path_to_data + '/' + scene_dir + '/parameters.cfg'
    config = configparser.RawConfigParser()
    config.read(path)
    baselineMM = config.getfloat('extrinsics', 'baseline_mm')
    focalLength = config.getfloat('intrinsics', 'focal_length_mm')
    sensorWidth = config.getfloat('intrinsics', 'sensor_size_mm')
    focus_distance_m = config.getfloat('extrinsics', 'focus_distance_m')


    return baselineMM, focalLength, sensorWidth, focus_distance_m


def get_target_image_from_disk(path_to_data, scene_dir, target_image_coords):
    
    path = path_to_data + '/' + scene_dir + '/' + 'input_Cam' + parse_coordinates2camsnumbering(target_image_coords) +'.png' 
    pil_image = Image.open(path)
    #pil_image = pil_image.resize((128, 128))
    #TODO implement resizing operation 
    pil2tensor = transforms.ToTensor()
    rgb_image = torch.squeeze(pil2tensor(pil_image))
    rgb_image.requires_grad = True

    return rgb_image
