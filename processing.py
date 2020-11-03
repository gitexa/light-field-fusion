from PIL.Image import alpha_composite
import torch
from torchvision import transforms
from torchvision import utils
import dataset_processing
import h5py
import pickle
import os
import get_data
import pickle

# Input  : (512,512) depth_map
#          Integer layers          
# Output : (512,512, layers) depth tensor, scalar min_disp by which the disp_tensor is shifted, scalar bin_size

def create_disp_tensor(depth_map, layers):
    assert depth_map.shape == (512,512), 'Depth map needs to be shape (512,512)'
    assert type(layers) == int, 'layers needs to be integer'

	# convert the depth to disparity
    disp_map = 1./(depth_map.float()+0.000001)


    #get the min_disparity and the max_disparity
    min_disp = torch.min(disp_map)*0.999999
    max_disp = torch.max(disp_map)*1.000001

    #get the bin_size
    bin_size  = (max_disp-min_disp)/layers

    #get the disparity_layer for every pixel
    disp_layers = layers - 1 - ((disp_map-min_disp)/bin_size).int()
    #convert to 3d 1-hot-encoding disparity_tensor (512,512,layers)
    disp_tensor = (torch.arange(layers) == disp_layers[...,None]).int()

    if(torch.isinf(bin_size)):
        bin_size = 0
        print('Bin size inf')
    
    return disp_tensor, min_disp, bin_size

# takes a 9x9 grid of depth_maps and returns a 9x9 grid of disp_tensors and 9x9 grids of min_disps and bin_sizes
def create_all_disp_tensors(depth_all_map, layers):
    assert depth_all_map.shape == (9,9,512,512), 'Depth_all_map needs to be 9x9x512x512'
    assert type(layers) == int, 'layers needs to be integer'
    
    all_disp_tensor = torch.zeros((9,9,512,512,layers))
    all_min_disp = torch.zeros((9,9))
    all_bin_size = torch.zeros((9,9))
    for i in range(9):
        for j in range(9):
            all_disp_tensor[i,j], all_min_disp[i,j], all_bin_size[i,j] = create_disp_tensor(depth_all_map[i,j],layers)

    return all_disp_tensor, all_min_disp, all_bin_size
    
def create_psv(image, disp_tensor, layers):
    
    # Input: RGB image 3 x 512 x 512 
    # Input: Disp tensor 512 x 512 x Depth
    # Output: PSV 3 x 512 x 512 x Depth
    
    assert len(disp_tensor.shape)==3, 'Disp tensor needs to be 512x512xdepth'
    layers = disp_tensor.shape[2]
    assert disp_tensor.shape == (512,512,layers), 'Depth tensor needs to be 512x512xdepth'

    assert len(image.shape)==3, 'Image must be 3x512x512'
    assert image.shape[0]==3, 'Image must be 3x512x512'

    output=torch.zeros((3,512,512,layers))
    
    for d in range(layers):
        output[:,:,:,d] = image*disp_tensor[None, :, :, d]
                
    return output

# takes a 9x9 grid of images and disp_tensors and returns a 9x9 grid of PSVs
def create_all_psv(image_all, disp_tensor_all, layers):
    all_psv = torch.zeros((9,9,3,512,512,layers))
    for i in range(9):
        for j in range(9):
            all_psv[i,j]=create_psv(image_all[i,j],disp_tensor_all[i,j], layers)
    return all_psv
    
    
def dataset_into_psvs(data_folder, layers=8):
    
    LF            = get_data.read_lightfield(data_folder)
    param         = get_data.read_parameters(data_folder)
    depth_all_map = get_data.read_all_depths(data_folder, highres=False)
    
    LF            = torch.from_numpy(LF).permute([0,1,4,2,3]).float()/255.
    depth_all_map = torch.from_numpy(depth_all_map)
    
    disp_tensors, min_disps, bin_sizes = create_all_disp_tensors(depth_all_map, layers)
    psvs = create_all_psv(LF, disp_tensors, layers)
    
    #return:
    # 9x9x3x512x512x8 PSV tensor
    # 9x9x1 Min_disp of every image
    # scene parameters
    return psvs, min_disps, bin_sizes, param

def save_psvs(psvs, min_disps, bin_sizes,param, data_folder, scene_number):
    
    for i in range(9):
        for j in range(9):
            coords = dataset_processing.parse_coordinates2camsnumbering((i,j))
            imgpath = str(scene_number)+ coords +'.pt'
            print(imgpath)
            path = os.path.join(data_folder, imgpath)

            psv_package = dict()
            psv_package['psv'] = psvs[i,j].clone()
            psv_package['min_disp'] = min_disps[i,j].clone()
            psv_package['bin_size'] = bin_sizes[i,j].clone()

            torch.save(psv_package, path)
            #torch.save( [psvs[i,j].clone(), min_disps[i,j].clone(), bin_sizes[i,j].clone(), param] , path )
	
	
# This function is the important one
# Input: 
# data_folder path, e.g. "Data/set1"
# scene number to put in the names, i.e. the i in "i62.pt"
# the number of layers, default set to 8

# The function reads in the 9x9x512x512 depth-layer tensor and the 9x9x3x512x512 image tensor
# It safes 81 .pt files containing data of the form [PSV, Min_disp, bin_size, param]
# These can be loaded individually with        psv, min_disp,bin_size, param = torch.load( ... the path ...)
def create_psv_dataset(data_folder, scene_number='psv_', layers=8):

    
    psvs, min_disps, bin_sizes, param = dataset_into_psvs(data_folder, layers)
    
    save_psvs( psvs, min_disps, bin_sizes, param, data_folder, scene_number)
    
    
    
# Function to get target_image from predicted MPIs by warping and alpha-compositing one image from each MPI in target view and blending them to one final target_image by equation 8 
# Input: prediced MPIs
# Output: target view 

def get_target_image(mpis, data):

    mpi_images = list()

    # warp every mpi slice into target perspective
    warped_mpis = homography(mpis, data)

    # split tensor in single mpis
    warped_mpis_split = torch.split(warped_mpis, split_size_or_sections=1, dim=0)
        
    # Back2front compositing
    for mpi in warped_mpis_split:
        composited_image = back_to_front_alphacomposite(torch.squeeze(mpi))
        mpi_images.append(composited_image)
    
    # blending rgba_images together to get target_image
    mpi_images = torch.squeeze(torch.stack(mpi_images, dim=0))
    target_image = blending_images_ourspecialcase(mpi_images)
    
    return target_image   
    
    
    
# takes a depth tensor of shape (512,512,depth) as input
# input is assumed to assign every pixel of the (512,512) image to exactly one depth
# output is a (512,512,depth) tensor with smoothed depth information
# smoothes depth channel with filter [0.1 , 0.2 , 0.4 , 0.2 , 0.1]

def naive_alpha(depth_tensor):
    
    assert len(depth_tensor.shape)==3, 'Depth tensor needs to be 512x512xdepth'
    depth = depth_tensor.shape[2]
    assert depth_tensor.shape == (512,512,depth), 'Depth tensor needs to be 512x512xdepth'
    #assert torch.allclose(torch.sum(depth_tensor, dim=2), torch.ones((512,512))), 'Depth tensor not normalized'
    
    # Note: Add 4 units of 0-padding in depth direction for the boundary conditions
    zero_padding = torch.zeros((512,512,2))
    alpha        = torch.cat((zero_padding, depth_tensor, zero_padding), dim=2)
    
    # Naively distribute the alpha=1 value over its neighborhood
    # Equivalent to a 1D-convolution of the depth dimension with the filter [0.1 , 0.2 , 0.4 , 0.2 , 0.1]  ??
    alpha = alpha[:,:,0:-4]*0.1 + alpha[:,:,1:-3]*0.2 + alpha[:,:,2:-2]*0.4 + alpha[:,:,3:-1]*0.2 + alpha[:,:,4:]*0.1

    
    # Assure normalization
    # Note: In case the significant depth is at the boundary, this is necessary ??
    sum_along_depth = torch.sum(alpha, dim=2)
    alpha           = alpha / sum_along_depth.unsqueeze(2)
    
    assert alpha.shape == (512,512,depth), 'Alpha Dimensionality failed'
    #assert torch.allclose(torch.sum(alpha, dim=2), torch.ones((512,512))), 'Alpha Normalization failed'

    
    return alpha
    
    
    
    
def rgba(image, depth_tensor):
    
    # Input: RGB image 3 x 512 x 512 
    # Input: Depth tensor 512 x 512 x Depth
    # Output: RGB-alpha depth image 4 x 512 x 512 x Depth
    
    assert len(depth_tensor.shape)==3, 'Depth tensor needs to be 512x512xdepth'
    Depth = depth_tensor.shape[2]
    assert depth_tensor.shape == (512,512,Depth), 'Depth tensor needs to be 512x512xdepth'
    #assert torch.allclose(torch.sum(depth_tensor, dim=2), torch.ones((512,512))), 'Depth tensor not normalized'
    
    assert len(image.shape)==3, 'Image must be 3x512x512'
    assert image.shape[0]==3, 'Image must be 3x512x512'

    
    output=torch.zeros((4,512,512,Depth))
    
    for d in range(Depth):
                
        output[:,:,:,d]=torch.stack([image[0,:,:],image[1,:,:],image[2,:,:],depth_tensor[:,:,d]], dim=0)
                
    return output
    
def rgba_center_psv(psv):

    # PSV dimension (2,3,512,512,depth)
    alpha = torch.zeros((2,1,512,512,8))
    height = 512
    depth = 512
    height = 8
    for h in range(height):
        for w in range(width): 
            for d in range(depth):
                if (psv_r[:,0,h,w,d] > 0):
                    alpha[:,0,h,w,d] = 1
    
    rgba = torch.stack((psvs, alpha), dim=1)

    return rgba

    
    
    
    
# Function to alpha_composite a 4d rgb-alpha depth image from back to front
# Input  (4,512,512,Depth)
# Output (4,512,512)
def back_to_front_alphacomposite(rgba_depth_image):
    
    assert len(rgba_depth_image.shape)==4 , 'Input image needs to have shape 4 x 512 x 512 x Depth'
    assert rgba_depth_image.shape[0] == 4 , 'Input image needs to have shape 4 x 512 x 512 x Depth'
    Depth = rgba_depth_image.shape[3] 

    mpiR_alpha = rgba_depth_image[..., 3:4] # 1 H W D 1
    mpiR_color = rgba_depth_image[..., 0:3] # 1 H W D 3
    
    img = rgba_depth_image[:,:,:,-1]


    for d in reversed(range(1,Depth)):
        
        layer = rgba_depth_image[:,:,:,d-1]

        alpha_new = layer[-1]+img[-1]*(1-layer[-1]) + 1e-10
        r_new     = (layer[-1]*layer[0]+img[-1]*img[0]*(1-layer[-1]))/alpha_new
        g_new     = (layer[-1]*layer[1]+img[-1]*img[1]*(1-layer[-1]))/alpha_new
        b_new     = (layer[-1]*layer[2]+img[-1]*img[2]*(1-layer[-1]))/alpha_new
                                  
       
        img = torch.stack([r_new, g_new, b_new, alpha_new], dim=0)

    return img

def back_to_front_alphacomposite_clone(rgba_depth_image):
    
    assert len(rgba_depth_image.shape)==4 , 'Input image needs to have shape 4 x 512 x 512 x Depth'
    assert rgba_depth_image.shape[0] == 4 , 'Input image needs to have shape 4 x 512 x 512 x Depth'
    Depth = rgba_depth_image.shape[3] 
    
    img = rgba_depth_image[:,:,:,-1].clone()


    for d in reversed(range(1,Depth)):
        
        layer = rgba_depth_image[:,:,:,d-1]

        

        alpha_new = layer[-1]+img[-1]*(1-layer[-1]) + 1e-10
        r_new     = (layer[-1]*layer[0]+img[-1]*img[0]*(1-layer[-1]))/alpha_new
        g_new     = (layer[-1]*layer[1]+img[-1]*img[1]*(1-layer[-1]))/alpha_new
        b_new     = (layer[-1]*layer[2]+img[-1]*img[2]*(1-layer[-1]))/alpha_new
                                  
       
        img = torch.stack([r_new, g_new, b_new, alpha_new], dim=0)

    return img


def back_to_front_alphacomposite_pillow(rgba_depth_image):
    
    assert len(rgba_depth_image.shape)==4 , 'Input image needs to have shape 4 x 512 x 512 x Depth'
    assert rgba_depth_image.shape[0] == 4 , 'Input image needs to have shape 4 x 512 x 512 x Depth'
    Depth = rgba_depth_image.shape[3] 
    
    img = transforms.ToPILImage('RGBA')(rgba_depth_image[:,:,:,-1])

    for d in reversed(range(1,Depth)):
        
        layer = rgba_depth_image[:,:,:,d-1]
        layer = transforms.ToPILImage('RGBA')(layer)
        
        img = alpha_composite( layer , img)

    img = transforms.ToTensor()(img)
    return img


# Function to get bilinear interpolation weights for target pose calculation from 4 camera poses (regular grid)
# Input: target pose x and y; 4 camera poses 
# Output: weights for bilinear interpolation 

def bilinear_interpolation(x, y, poses):

    poses = sorted(poses)               
    (x1, y1), (_x1, y2), (x2, _y1), (_x2, _y2) = poses

    w1 = (x2 - x) * (y2 - y) / ((x2 - x1) * (y2 - y1) + 0.0)
    w2 = (x - x1) * (y2 - y) / ((x2 - x1) * (y2 - y1) + 0.0)
    w3 = (x2 - x) * (y - y1) / ((x2 - x1) * (y2 - y1) + 0.0)
    w4 = (x - x1) * (y - y1) / ((x2 - x1) * (y2 - y1) + 0.0)

    return w1, w2, w3, w4

# Function to render target pose using alpha and rgb image and interpolation weights 
# Input: alpha- and rgb-images (MPI), target pose, camera poses
# Output: rendered target view

def render_target_view(alphas, rgbs, p_target, poses):
    


    w_t_1, w_t_2, w_t_3, w_t_4 = bilinear_interpolation(p_target[0], p_target[1], poses)

    r = (w_t_1 * torch.mul(alphas[0], rgbs[0,0,:,:]) + w_t_2 * torch.mul(alphas[1], rgbs[1,0,:,:]) + w_t_3 * torch.mul(alphas[2], rgbs[2,0,:,:]) + w_t_4 * torch.mul(alphas[3], rgbs[3,0,:,:])) / (w_t_1*alphas[0] + w_t_2*alphas[1] + w_t_3*alphas[2] + w_t_4*alphas[3])
    g = (w_t_1 * torch.mul(alphas[0], rgbs[0,1,:,:]) + w_t_2 * torch.mul(alphas[1], rgbs[1,1,:,:]) + w_t_3 * torch.mul(alphas[2], rgbs[2,1,:,:]) + w_t_4 * torch.mul(alphas[3], rgbs[3,1,:,:])) / (w_t_1*alphas[0] + w_t_2*alphas[1] + w_t_3*alphas[2] + w_t_4*alphas[3])
    b = (w_t_1 * torch.mul(alphas[0], rgbs[0,2,:,:]) + w_t_2 * torch.mul(alphas[1], rgbs[1,2,:,:]) + w_t_3 * torch.mul(alphas[2], rgbs[2,2,:,:]) + w_t_4 * torch.mul(alphas[3], rgbs[3,2,:,:])) / (w_t_1*alphas[0] + w_t_2*alphas[1] + w_t_3*alphas[2] + w_t_4*alphas[3])

    target_view = torch.stack((r,g,b), dim=0)

    return target_view

#target_view = render_target_view(alphas, rgbs, p_target, poses)
#print(target_view)





#F(mm) = F(pixels) * SensorWidth(mm) / ImageWidth (pixel).

# homography warp
# http://campar.in.tum.de/twiki/pub/Chair/TeachingWs10Cv2/3D_CV2_WS_2010_Rectification_Disparity.pdf

# d = f*b / Z
# 1/Z given
# b baseline [mm]
# f focal length [mm]


# function to perform the homography warp 

def homography(mpis, input_dict):
  
    #mpi1 = input_dict['psvs'][0]
    #mpi2 = input_dict['psvs'][1]
    mpi1 = mpis[0]
    mpi2 = mpis[1]
    layers = mpi1.shape[-1]

    
    baseline = input_dict['baselineMM']
    focal_length = input_dict['focalLength']
    sensor_size = float(input_dict['sensorWidthMM'])
    min_disp1 = input_dict['mpi_1_min_disp'] 
    min_disp2 = input_dict['mpi_2_min_disp'] 
    bin_size1 = input_dict['mpi_1_bin_size'] 
    bin_size2 = input_dict['mpi_2_bin_size'] 
    focus_dist = float(input_dict["focus_distance_m"])
    
    disparity_factor = focal_length * baseline * (512./sensor_size) / 1000.

    mpi1_pos = input_dict['psv_center_1_pose']
    mpi2_pos = input_dict['psv_center_2_pose']
    target_pos = input_dict['target_image_pose']
    
    #warp the first image
    target_mpi1 = torch.zeros(mpi1.shape)
    target_mpi2 = torch.zeros(mpi2.shape)
    
    camera_xDiff1 = (mpi1_pos[0]-target_pos[0])
    camera_yDiff1 = (mpi1_pos[1]-target_pos[1])
    camera_xDiff2 = (mpi2_pos[0]-target_pos[0])
    camera_yDiff2 = (mpi2_pos[1]-target_pos[1])

    if camera_xDiff1 != 0:
        for d in range(layers):
            
            
            xdisparity1 = int(((layers-1-d)*bin_size1 + min_disp1 - (1/focus_dist))*disparity_factor*camera_xDiff1)
            xdisparity2 = int(((layers-1-d)*bin_size2 + min_disp2 - (1/focus_dist))*disparity_factor*camera_xDiff2)
            

        
            if xdisparity1 >0:
                target_mpi1[:,xdisparity1:,:,d] = mpi1[:,:-xdisparity1,:,d]
            elif xdisparity1 < 0:
                
                
               
                target_mpi1[:,:xdisparity1,:,d] = mpi1[:,-xdisparity1:,:,d]
            else:
                target_mpi1[:,:,:,d]=mpi1[:,:,:,d]
               
            if xdisparity2 >0:
                target_mpi2[:,xdisparity2:,:,d] = mpi2[:,:-xdisparity2,:,d]
            elif xdisparity2 < 0:
                
                
                
                target_mpi2[:,:xdisparity2,:,d] = mpi2[:,-xdisparity2:,:,d]
            else:
                target_mpi2[:,:,:,d]=mpi2[:,:,:,d]
        return torch.stack([target_mpi1, target_mpi2] , dim=0)
    
    else:
        for d in range(layers):

            ydisparity1 = int(((layers-1-d)*bin_size1 + min_disp1 - (1/focus_dist))*disparity_factor*camera_yDiff1)
            ydisparity2 = int(((layers-1-d)*bin_size2 + min_disp2 - (1/focus_dist))*disparity_factor*camera_yDiff2)

            if ydisparity1 >0:
                target_mpi1[:,:,ydisparity1:,d] = mpi1[:,:,:-ydisparity1,d]
            elif ydisparity1 < 0:
                
                target_mpi1[:,:,:ydisparity1,d] = mpi1[:,:,-ydisparity1:,d]
            else:
                target_mpi1[:,:,:,d]=mpi1[:,:,:,d]
                
            if ydisparity2 >0:
                target_mpi2[:,:,ydisparity2:,d] = mpi2[:,:,:-ydisparity2,d]
            elif ydisparity2 < 0:
                
                target_mpi2[:,:,:ydisparity2,d] = mpi2[:,:,-ydisparity2:,d]
            else:
                target_mpi2[:,:,:,d]=mpi2[:,:,:,d]
        return torch.stack([target_mpi1, target_mpi2] , dim=0)

    
def blending_images_ourspecialcase(rgba):

    w_t_1 = 0.5
    w_t_2 = 0.5

    r = (w_t_1 * torch.mul(rgba[0][3], rgba[0][0]) + w_t_2 * torch.mul(rgba[1][3], rgba[1][0])) / (w_t_1*rgba[0][3] + w_t_2*rgba[1][3])
    g = (w_t_1 * torch.mul(rgba[0][3], rgba[0][1]) + w_t_2 * torch.mul(rgba[1][3], rgba[1][1])) / (w_t_1*rgba[0][3] + w_t_2*rgba[1][3])
    b = (w_t_1 * torch.mul(rgba[0][3], rgba[0][2]) + w_t_2 * torch.mul(rgba[1][3], rgba[1][2])) / (w_t_1*rgba[0][3] + w_t_2*rgba[1][3])

    target_view = torch.squeeze(torch.stack((r,g,b), dim=0))

    return target_view

def save_images(relative_path_to_results, target_image, predicted_target_image, sample_id, target_image_pose, epoch, loss):
    path_target_image = relative_path_to_results + '/images/' + str(sample_id) + '_target_pose_' + str(target_image_pose) + '_epoch' + str(epoch) + '_target_image.png'
    path_predicted_image = relative_path_to_results + '/images/' + str(sample_id) + '_target_pose_' + str(target_image_pose) + '_epoch_' + str(epoch) + '_predicted_image_loss_' + str(loss) + '.png'

    utils.save_image(tensor=target_image, fp=path_target_image)
    utils.save_image(tensor=predicted_target_image, fp=path_predicted_image)

        
        
        

# die gute    
def network_into_mpi(tensor, psvs, device):
    
    #assume tensor is shape (2, 5,512,512,8)
    #assume psvs is shape (2,5,3,512,512,8)


    softmax_input = torch.stack([torch.stack([torch.zeros((512,512,8)).to(device), tensor[0,1], tensor[0,2], tensor[0,3], tensor[0,4]],dim=0),
                                 torch.stack([torch.zeros((512,512,8)).to(device), tensor[1,1], tensor[1,2], tensor[1,3], tensor[1,4]],dim=0)] ,dim=0)
    softmax_output = torch.nn.functional.softmax(softmax_input, dim=1)

    r = torch.sum(psvs[:,:,0,:,:,:]*softmax_output, dim=1)
    g = torch.sum(psvs[:,:,1,:,:,:]*softmax_output, dim=1)
    b = torch.sum(psvs[:,:,2,:,:,:]*softmax_output, dim=1)

    mpis = torch.stack( [r,g,b, torch.sigmoid(tensor[:,0])], dim=1)

    return mpis

# äquivalent zur ersten, etwas langsamer
def network_into_mpi2(tensor, psvs):
    
    #assume tensor is shape (2, 5,512,512,8)
    #assume psvs is shape (2,5,3,512,512,8)


    softmax_input1 = torch.stack( [torch.zeros((512,512,8)), tensor[0,1], tensor[0,2], tensor[0,3], tensor[0,4]], dim=0)
    softmax_output1 = torch.nn.functional.softmax(softmax_input1, dim=0)
    
    softmax_input2 = torch.stack( [torch.zeros((512,512,8)), tensor[1,1], tensor[1,2], tensor[1,3], tensor[1,4]], dim=0)
    softmax_output2 = torch.nn.functional.softmax(softmax_input2, dim=0)

    r1 = torch.sum(psvs[0,:,0,:,:,:]*softmax_output1, dim=0)
    g1 = torch.sum(psvs[0,:,1,:,:,:]*softmax_output1, dim=0)
    b1 = torch.sum(psvs[0,:,2,:,:,:]*softmax_output1, dim=0)
    
    r2 = torch.sum(psvs[1,:,0,:,:,:]*softmax_output2, dim=0)
    g2 = torch.sum(psvs[1,:,1,:,:,:]*softmax_output2, dim=0)
    b2 = torch.sum(psvs[1,:,2,:,:,:]*softmax_output2, dim=0)
    
    mpis = torch.stack( [torch.stack([r1,g1,b1,torch.sigmoid(tensor[0,0])], dim=0), torch.stack([r2,g2,b2,torch.sigmoid(tensor[1,0])], dim=0)], dim=0)
                       
    return mpis

def save_ckp(state, path):
    torch.save(state, path)
    
def load_ckp(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
