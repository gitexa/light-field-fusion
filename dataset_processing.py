'Helper Methods'


def parse_camsnumbering2coordinates(number):

    int_num = 51
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

        
