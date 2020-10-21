
import torch
import processing
import get_data
import small_net
import large_net
import matplotlib.pyplot as plt
import numpy as np



def get_psvs(LF, depth_all_map, layers):
    # convert the data to torch tensors
    images = torch.from_numpy(LF).permute([0,1,4,2,3]).float()/255.
    depth_maps = torch.from_numpy(depth_all_map)

    # create the depth tensors (PSV)
    psvs = torch.zeros((9,9,512,512,layers))
    for i in range(9):
        for j in range(9):
            psvs[i,j] = processing.create_depth_tensor(depth_maps[i,j], layers)
    
    return psvs, images

def get_mpis(psvs, images, layers):
    # convert the 0-1 depth tensor to smoothed alpha depth tensors
    alpha_tensors = torch.zeros((9,9,512,512,layers))
    for i in range(9):
        for j in range(9):
            alpha_tensors[i,j] = processing.naive_alpha(psvs[i,j])

    # convert alpha tensors and rgb images to rgba depth tensors
    rgba = torch.zeros((9,9,4,512,512,layers))
    for i in range(9):
        for j in range(9):
            rgba[i,j] = processing.rgba(images[i,j], alpha_tensors[i,j])

    return rgba

def transform_mpis(rgba):

    # Homography warp 
    # @TODO

    # alpha composite rgba images from the rgba depth tensors
    C_alpha = torch.zeros((9,9,4,512,512))
    for i in range(9):
        for j in range(9):
            C_alpha[i,j] = processing.back_to_front_alphacomposite(rgba[i,j])

    # seperate RGB images and alpha images
    rgb    = C_alpha[:,:,:3,:,:]
    alpha  = C_alpha[:,:, 3,:,:]

    return rgb, alpha 


def render_image(alpha, rgb, target_pose, input_poses):

    # select cameras 
    #alphas = alpha[:2, :2, :, :].reshape(-1, 512, 512)
    #rgbs = rgb[:2, :2, :, :].reshape(-1, 3, 512, 512)

    # render the new view 
    target_view = processing.render_target_view(alpha, rgb, target_pose, input_poses)
    
    return target_view


'''
Test pipeline 

'''

# read in the data
data_folder = "less_data/0cC7GPRFAIvP5i/"
LF = get_data.read_lightfield(data_folder)
param_dict = get_data.read_parameters(data_folder)
depth_map = get_data.read_depth(data_folder, highres=False)
depth_all_map = get_data.read_all_depths(data_folder, highres=False)

# Config
layers = 16 

# Fake camera poses 
target_pose = (1.5,1.7)
p_1 = (0,0)
p_2 = (0,1)
p_3 = (1,0)
p_4 = (1,1)
input_poses = list()
input_poses.append(p_1)
input_poses.append(p_2)
input_poses.append(p_3)
input_poses.append(p_4)

psvs, images = get_psvs(LF, depth_all_map, layers)

# mpi = get_mpis(psvs, images, layers)
# rgb, alpha = transform_mpis(mpi)

psv_1 = torch.rand((4,15,64,64,8))
psv_2 = torch.rand((4,15,64,64,8))
psv_3 = torch.rand((4,15,64,64,8))
psv_4 = torch.rand((4,15,64,64,8))
psv_5 = torch.rand((4,15,64,64,8))

psvs = list()

psvs.append(psv_1)
psvs.append(psv_2)
psvs.append(psv_3)
psvs.append(psv_4)
psvs.append(psv_5)


#model = small_net.MPIPredictionNet()
model = large_net.MPIPredictionNet()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=2e-4, amsgrad=True)

# Starting training pipeline
model.train()

for psv in psvs:
    
    optimizer.zero_grad()

    mpis = model(psv)

    target_view = torch.rand((3,64,64), requires_grad=True)
    ground_truth_target_view = torch.rand((3,64,64), requires_grad=True)

    loss = loss_function(target_view, ground_truth_target_view)

    loss.backward()

    optimizer.step()
    
#mpi_separated = torch.split(mpis, split_size_or_sections=1, dim=1)
#mpi_alpha = mpi_separated[0]
#mpi_alpha = torch.unsqueeze(torch.squeeze(mpi_alpha),dim=1)
#mpi_rgb = torch.squeeze(torch.stack((mpi_separated[1], mpi_separated[2], mpi_separated[3]), dim=1))

#processing.back_to_front_alphacomposite(mpis)

#target_view = render_image(mpi_alpha, mpi_rgb, target_pose, input_poses)

# print images
#original_img  = images[0,0].permute(1,2,0).numpy()
#reproduce_img = target_view.permute(1,2,0).numpy()

#plt.imshow(original_img)
#plt.show()
#plt.imshow(reproduce_img)
#plt.show()
