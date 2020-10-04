from PIL.Image import alpha_composite
import torch
from torchvision import transforms


# Input  : (512,512) depth_map
#          Integer layers          
# Output : (512,512, layers) depth tensor

def create_depth_tensor(depth_map, layers):
    assert depth_map.shape == (512,512), 'Depth map needs to be shape (512,512)'
    assert type(layers) == int, 'layers needs to be integer'
    
    #get the min_depth and the max_depth
    min_depth = torch.min(depth_map)
    max_depth = torch.max(depth_map)*1.000001
    
    #shift all depths by the min_depth
    depth_rel = depth_map-min_depth
    
    #get the bin_size
    bin_size  = (max_depth-min_depth)/layers
    
    #get the depth_layer for every pixel
    depth_layers = (depth_rel/bin_size).int()
    
    #convert to 3d 1-hot-encoding depth_tensor (512,512,layers)
    depth_tensor = (torch.arange(layers) == depth_layers[...,None]).int()
    
    return depth_tensor 
    
    
    
    
# takes a depth tensor of shape (512,512,depth) as input
# input is assumed to assign every pixel of the (512,512) image to exactly one depth
# output is a (512,512,depth) tensor with smoothed depth information
# smoothes depth channel with filter [0.1 , 0.2 , 0.4 , 0.2 , 0.1]

def naive_alpha(depth_tensor):
    
    assert len(depth_tensor.shape)==3, 'Depth tensor needs to be 512x512xdepth'
    depth = depth_tensor.shape[2]
    assert depth_tensor.shape == (512,512,depth), 'Depth tensor needs to be 512x512xdepth'
    assert torch.allclose(torch.sum(depth_tensor, dim=2), torch.ones((512,512))), 'Depth tensor not normalized'
    
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
    assert torch.allclose(torch.sum(alpha, dim=2), torch.ones((512,512))), 'Alpha Normalization failed'

    
    return alpha
    
    
    
    
def rgba(image, depth_tensor):
    
    # Input: RGB image 3 x 512 x 512 
    # Input: Depth tensor 512 x 512 x Depth
    # Output: RGB-alpha depth image 4 x 512 x 512 x Depth
    
    assert len(depth_tensor.shape)==3, 'Depth tensor needs to be 512x512xdepth'
    Depth = depth_tensor.shape[2]
    assert depth_tensor.shape == (512,512,Depth), 'Depth tensor needs to be 512x512xdepth'
    assert torch.allclose(torch.sum(depth_tensor, dim=2), torch.ones((512,512))), 'Depth tensor not normalized'
    
    assert len(image.shape)==3, 'Image must be 3x512x512'
    assert image.shape[0]==3, 'Image must be 3x512x512'

    
    output=torch.zeros((4,512,512,Depth))
    
    for d in range(Depth):
                
        output[:,:,:,d]=torch.stack([image[0,:,:],image[1,:,:],image[2,:,:],depth_tensor[:,:,d]], dim=0)
                
    return output
    
    
    
    
# Function to alpha_composite a 4d rgb-alpha depth image from back to front
# Input  (4,512,512,Depth)
# Output (4,512,512)

def back_to_front_alphacomposite(rgba_depth_image):
    
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





