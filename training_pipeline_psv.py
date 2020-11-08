import os
import dataset_processing
import dataset
import torch 
import processing
import RandomSampler
import get_data
from models.net_direct_mpi import MPIPredictionNet as MPIPredictionNet_directMPI
from models.net_weighted_mpi import MPIPredictionNet as MPIPredictionNet_weightedMPI
import matplotlib.pyplot as plt
import numpy as np
import gc
import json
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler


'Print method evokation'
print('-' * 60)
print('Started pipeline for approach with PSVs')
print('-' * 60)
print('Cuda?: ' + str(torch.cuda.is_available()))
if(torch.cuda.is_available() == True):
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
torch.manual_seed(0)
np.random.seed(0)

'Parameters'
#torch.manual_seed(0)
relative_path_to_results = 'results_psvs'
relative_path_to_scenes = '/media/mkg/Elements/03_MLData/lightfields/all_lightfields'
#relative_path_to_scenes = '/media/alexander/Elements/03_MLData/lightfields/all_lightfields'
validation_split = .2
layers = 8

'Create folder structure'
relative_path_to_results
os.path.exists(relative_path_to_results) 
if(not os.path.isdir(relative_path_to_results)):
    os.mkdir(relative_path_to_results)
    os.mkdir(relative_path_to_results + '/model')
    os.mkdir(relative_path_to_results + '/metrics')
    os.mkdir(relative_path_to_results + '/plots')
    os.mkdir(relative_path_to_results + '/images')
assert os.path.isdir(relative_path_to_scenes)


'Read all scenes and generate all ids'
#TODO add all scenes from dataset 
#all_scenes = list()
#all_scenes.append('0cC7GPRFAIvP5i')
#all_scenes.append('1eTVjMYXkOBq6b')
#all_scenes.append('1eTVjMYXkOBq6b')
all_scenes = dataset_processing.get_all_scenes(relative_path_to_scenes)
#all_scenes = ['gZ392ME3DDQPeX']
all_ids = dataset_processing.generate_all_ids(all_scenes)
num_scenes = len(all_scenes)

'Create PSV dataset (only once necessary)' 
#Only once!!!
#for scene in all_scenes:
#    processing.create_psv_dataset(relative_path_to_scenes + '/' + scene, layers=layers)

'Create customized pytorch dataset'
random_seed = 42
all_data = dataset.Dataset(all_ids, relative_path_to_scenes, layers)
all_data_size = len(all_data)
indices = list(range(all_data_size))
split = int(np.floor(validation_split * all_data_size))
#np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
#train_sampler = RandomSampler.RandomSampler(train_indices)
#valid_sampler = RandomSampler.RandomSampler(val_indices)
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
training_generator = torch.utils.data.DataLoader(all_data, batch_size=1, sampler=train_sampler, num_workers=10)
validation_generator = torch.utils.data.DataLoader(all_data, batch_size=1, sampler=valid_sampler, num_workers=10)

'Create loss function'
loss_function = torch.nn.MSELoss()

'Loss for training and validation'
validation_epoch_loss_values = list()
best_metric_epoch = 0

'Start validation process'
print('-' * 60)
print(f"Validation")
val_loss = 0
val_step = 0

for data in validation_generator:
    val_step += 1
    print('Sample: ' + data['sample_id'][0])

    # Get target_image
    target_image = torch.squeeze(data['target_image'])

    'Calculate RGBA-PSVs and prediction for target view'
    # Get files
    data_folder = relative_path_to_scenes + '/' + str(data['scene_id'][0])
    LF            = get_data.read_lightfield(data_folder)
    param_dict    = get_data.read_parameters(data_folder)
    depth_map     = get_data.read_depth(data_folder, highres=False)
    depth_all_map = get_data.read_all_depths(data_folder, highres=False)

    # Convert the data to torch tensors
    images     = torch.from_numpy(LF).permute([0,1,4,2,3]).float()/255.
    depth_maps = torch.from_numpy(depth_all_map)

    # Get picuture and corresponding depth map
    image_1 = images[data['psv_center_1_pose'][0], data['psv_center_1_pose'][1]]
    depth_map_1 = depth_maps[data['psv_center_1_pose'][0], data['psv_center_1_pose'][1]]
    image_2 = images[data['psv_center_2_pose'][0], data['psv_center_2_pose'][1]]
    depth_map_2 = depth_maps[data['psv_center_2_pose'][0], data['psv_center_2_pose'][1]]
    
    # Get disp tenspr
    disp_tensor_1 = processing.create_disp_tensor(torch.squeeze(depth_map_1), layers)[0]
    disp_tensor_2 = processing.create_disp_tensor(torch.squeeze(depth_map_2), layers)[0]

    # Get mpi
    psv_mpi1 = processing.rgba(torch.squeeze(image_1), disp_tensor_1)
    psv_mpi2 = processing.rgba(torch.squeeze(image_2), disp_tensor_2)

    # Homography warp
    mpis = torch.stack([psv_mpi1, psv_mpi2], dim=0)
    target_mpis = processing.homography(mpis, data)

    # jetzt das alpha compositing
    rgba1 = processing.back_to_front_alphacomposite(target_mpis[0])
    rgba2 = processing.back_to_front_alphacomposite(target_mpis[1])

    # jetzt das rendering als superposition der beiden
    predicted_image = processing.blending_images_ourspecialcase(torch.stack((rgba1, rgba2), dim=0))

    if(torch.cuda.is_available() == True):
        predicted_image, target_image = predicted_image.to(device), target_image.to(device)
    
    loss = loss_function(target_image, predicted_image)
    processing.save_images(relative_path_to_results, target_image, predicted_image, data['sample_id'][0], dataset_processing.coords2string((data['target_image_pose'][0].item(), data['target_image_pose'][1].item())), 0, loss.item())
    val_loss += loss.item()

    print(f"{val_step}/{len(validation_generator) // validation_generator.batch_size}, val_loss: {loss.item():.4f}")

   
val_loss /= val_step
#validation_epoch_loss_values.append((epoch, val_loss))

print(f"Summary | Validation | Average loss {val_loss:.4f}")
print("All metrics and images saved")
print(f"Current metric {val_loss:.4f}")

    
'Save metrics'
with open(relative_path_to_results + '/metrics/' + 'validation_psvonly_loss_values.txt', 'w') as filehandle:
    json.dump(val_loss, filehandle)