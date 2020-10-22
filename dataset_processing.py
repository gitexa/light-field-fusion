import torch

relative_path_to_scenes = 'less_data/'


def get_nn(sample_mpi_poses):
    
    # get psvs of first mpi_pose
    psv_1_center = (sample_mpi_poses[0][0], sample_mpi_poses[0][0])
    psv_1_ul = (sample_mpi_poses[0][0]-1, sample_mpi_poses[0][0]-1)
    psv_1_ur = (sample_mpi_poses[0][0]-1, sample_mpi_poses[0][0]+1)
    psv_1_ll = (sample_mpi_poses[0][0]+1, sample_mpi_poses[0][0]-1)
    psv_1_lr = (sample_mpi_poses[0][0]+1, sample_mpi_poses[0][0]+1)

    # get psvs of second mpi_pose
    psv_2_center = (sample_mpi_poses[1][0], sample_mpi_poses[1][0])
    psv_2_ul = (sample_mpi_poses[1][0]-1, sample_mpi_poses[1][0]-1)
    psv_2_ur = (sample_mpi_poses[1][0]-1, sample_mpi_poses[1][0]+1)
    psv_2_ll = (sample_mpi_poses[1][0]+1, sample_mpi_poses[1][0]-1)
    psv_2_lr = (sample_mpi_poses[1][0]+1, sample_mpi_poses[1][0]+1)

    # get psv tensors from file 
    psvs_1 = torch.stack((psv_1_center, psv_1_ul, psv_1_ur, psv_1_ll, psv_1_lr), dim=0)
    psvs_2 = torch.stack((psv_2_center, psv_2_ul, psv_2_ur, psv_2_ll, psv_2_lr), dim=0)

    psvs = torch.squeeze(torch.stack((psvs_1, psvs_2), dim=0))
    
    return psvs
    





# Function to get 5 PSVS for each of 2 MPI target poses 
def get_nn_psvs(row, column):

    first_mpi_pose = (row, column)
    
    # select second MPI pose
    if(row<=3 and column <=3):
        second_mpi_pose_var1 = (row, column+3)
        second_mpi_pose_var2 = (column, row+3)
    elif(row<=3 and column>3):
        second_mpi_pose_var1 = (row, column-3)
        second_mpi_pose_var2 = (column, row+3)
    elif(row>3 and column<=3):
        second_mpi_pose_var1 = (row, column+3)
        second_mpi_pose_var2 = (column, row-3)
    elif(row>3 and column>3):
        second_mpi_pose_var1 = (row, column-3)
        second_mpi_pose_var2 = (column, row-3)

    sample_1 = (first_mpi_pose, second_mpi_pose_var1)
    sample_2 = (first_mpi_pose, second_mpi_pose_var2)

    nn_sample_1 = get_nn(sample_1)
    nn_sample_2 = get_nn(sample_2)

    return sample


w, h = 9, 9;
all_views = [[0 for x in range(w)] for y in range(h)] 

for row in range(9):
    for column in range(9):
        allviews[row][column] = get_nn_psvs(row, column)

        
