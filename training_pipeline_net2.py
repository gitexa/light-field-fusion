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
print('Started pipeline')
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
relative_path_to_results = 'results_approach2'
relative_path_to_scenes = '/media/mkg/Elements/03_MLData/lightfields/all_lightfields'
relative_path_to_checkpoint = 'results_approach2/model/epoch_3training_best_metric_model.pth'
max_epochs = 250
validation_split = .2
val_interval = 5
layers = 8
#image_size = (64,64)
load_checkpoint = True
load_indices = True


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
#all_scenes = ['qPS9zDEjhwzIez']
all_ids = dataset_processing.generate_all_ids(all_scenes)
num_scenes = len(all_scenes)


'Create PSV dataset (only once necessary)' 
# Only once!!!
#for scene in all_scenes:
#    processing.create_psv_dataset(relative_path_to_scenes + '/' + scene, layers=layers)


'Create customized pytorch dataset'
if (load_indices == True):
    random_seed = 42
    all_data = dataset.Dataset(all_ids, relative_path_to_scenes, layers)
    with open(relative_path_to_results + '/train_indices.json') as f:
        train_indices = json.load(f)
    with open(relative_path_to_results + '/val_indices.json') as f:
        val_indices = json.load(f)        
else:
    random_seed = 42
    all_data = dataset.Dataset(all_ids, relative_path_to_scenes, layers)
    all_data_size = len(all_data)
    indices = list(range(all_data_size))
    split = int(np.floor(validation_split * all_data_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

#train_sampler = RandomSampler.RandomSampler(train_indices)
#valid_sampler = RandomSampler.RandomSampler(val_indices)
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
training_generator = torch.utils.data.DataLoader(all_data, batch_size=1, sampler=train_sampler)
validation_generator = torch.utils.data.DataLoader(all_data, batch_size=1, sampler=valid_sampler)

with open(relative_path_to_results + '/train_indices.json', 'w') as filehandle:
    json.dump(train_indices, filehandle)
with open(relative_path_to_results + '/val_indices.json', 'w') as filehandle:
    json.dump(val_indices, filehandle)

'Create model, loss function and optimizer'
if (load_checkpoint == True):
    # Restore model, optimizer, epoch
    model = MPIPredictionNet_weightedMPI(device)
    if(torch.cuda.is_available() == True):
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=2e-4, amsgrad=True)
    model_state_dict, optimizer_state_dict, epoch = processing.load_ckp(relative_path_to_checkpoint)
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    model_state_dict, optimizer_state_dict, epoch = processing.load_ckp(relative_path_to_checkpoint)
    loss_function = torch.nn.MSELoss()
    # Restore losses
    with open(relative_path_to_results + '/metrics/' + 'training_epoch_loss_values.txt') as f:
        training_epoch_loss_values = json.load(f)
    with open(relative_path_to_results + '/metrics/' + 'validation_epoch_loss_values.txt') as f:
        validation_epoch_loss_values = json.load(f)
    
    best_training_metric = min([tupl[1] for tupl in training_epoch_loss_values])
    best_training_metric_epoch = training_epoch_loss_values[[tupl[1] for tupl in training_epoch_loss_values].index(best_training_metric)][0] # get minimum training loss and corresponding epoch

    best_metric =  min([tupl[1] for tupl in validation_epoch_loss_values])
    best_metric_epoch = validation_epoch_loss_values[[tupl[1] for tupl in validation_epoch_loss_values].index(best_metric)][0] # get minimum training loss and corresponding epoch


else:
    # Define models
    #model = large_net.MPIPredictionNet()
    #model = MPIPredictionNet_directMPI()
    model = MPIPredictionNet_weightedMPI(device)
    if(torch.cuda.is_available() == True):
        model.to(device)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=2e-4, amsgrad=True)
    # Define losses 
    training_epoch_loss_values = list()
    validation_epoch_loss_values = list()
    best_metric = 100 #TODO random value, to be improved
    best_metric_epoch = 0
    best_training_metric = 100
    best_training_metric_epoch = 0 

'Collect corrupted ids'
corrupted_ids = list()

'Start training process'
print('-' * 60)
print('Start training:')
for epoch in range(epoch, max_epochs):
    print('-' * 60)
    print(f"Training | Epoch {epoch + 1}/{max_epochs}")
    epoch_loss = 0
    step = 0
    model.train()
    if(torch.cuda.is_available() == True):
        torch.cuda.empty_cache()

    'Forward and backward in batches (batches with 2x5 PSVs are assembled inside dataloader, batch_size therefore 1'
    for data in training_generator:

        with torch.autograd.set_detect_anomaly(True):

            # Get quick and dirty rid of errors we just discovered in the lightfield dataset
            if((data['mpi_1_min_disp']>1/1000) or (data['mpi_2_min_disp']>1/1000)):
                step += 1
                print('Sample: ' + data['sample_id'][0])
                optimizer.zero_grad()
                psvs, target_image = torch.squeeze(data['psvs']), torch.squeeze(data['target_image'])
                if(torch.cuda.is_available() == True):
                    psvs, target_image = psvs.to(device), target_image.to(device)

                mpis = model(psvs)
                
                predicted_image = processing.get_target_image(mpis, data)
                if(torch.cuda.is_available() == True):
                    predicted_image = predicted_image.to(device)
                
                #plt.imshow(  predicted_image.to("cpu").permute(1, 2, 0)  )
                #plt.savefig(relative_path_to_results + '/' + 'pred.png')

                #plt.imshow(  target_image.to("cpu").detach().permute(1, 2, 0)  )
                #plt.savefig(relative_path_to_results + '/' + 'targ.png')

                #model_params = list()
                #for name, param in model.named_parameters():
                #    model_params.append((name, param.data))

                loss = loss_function(target_image, predicted_image)
                loss.backward()
                optimizer.step()

                #model_params_after_grad = list()
                #for name, param in model.named_parameters():
                #    model_params_after_grad.append((name, param.data))
                
                #print(torch.max(model_params[0][1]-model_params_after_grad[0][1]))

                epoch_loss += loss.item()
                print(f"{step}/{len(training_generator) // training_generator.batch_size}, train_loss: {loss.item():.4f}")

                if(torch.cuda.is_available() == True):
                    torch.cuda.empty_cache()
                    #del variables 
                    #gc.collect()

            else:
                corrupted_ids.append(data['sample_id'])

                #TODO
                #if(step>10):
                #    break
            
            #break

    epoch_loss /= step
    training_epoch_loss_values.append((epoch, epoch_loss))
    print('-' * 10)
    print(f"Summary | Training | Epoch {epoch + 1}/{max_epochs} | Average loss {epoch_loss:.4f}")


    if(epoch_loss<best_training_metric):

        best_training_metric = epoch_loss
        best_training_metric_epoch = epoch

        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        processing.save_ckp(checkpoint, relative_path_to_results + '/model/epoch_'+ str((epoch)) +'training_best_metric_model.pth')

        print('Saved new best metric model in training.')



    'Validate for every val_interval epoch'
    if (epoch == 0) or ((epoch + 1) % val_interval == 0):        
        model.eval()
        with torch.no_grad():
            print('-' * 60)
            print(f"Validation | Epoch {epoch + 1}/{max_epochs}")
            val_loss = 0
            val_step = 0
            for data in validation_generator:
                with torch.autograd.set_detect_anomaly(True):
                    # Get quick and dirty rid of errors we just discovered in the lightfield dataset
                    if((data['mpi_1_min_disp']>1/1000) or (data['mpi_2_min_disp']>1/1000)):
                        val_step += 1
                        print('Sample: ' + data['sample_id'][0])
                        psvs, target_image = torch.squeeze(data['psvs']), torch.squeeze(data['target_image'])
                        if(torch.cuda.is_available() == True):
                            psvs, target_image = psvs.to(device), target_image.to(device)

                        mpis = model(psvs)

                        predicted_image = processing.get_target_image(mpis, data)
                        if(torch.cuda.is_available() == True):
                            predicted_image = predicted_image.to(device)
                        
                        loss = loss_function(target_image, predicted_image)
                        processing.save_images(relative_path_to_results, target_image, predicted_image, data['sample_id'][0], dataset_processing.coords2string((data['target_image_pose'][0].item(), data['target_image_pose'][1].item())), epoch, loss.item())
                        val_loss += loss.item()
                        print(f"{val_step}/{len(validation_generator) // validation_generator.batch_size}, val_loss: {loss.item():.4f}")
                    
                    else:
                        pass

                    #TODO
                    #if(val_step>10):
                    #    break
                    #break 

            val_loss /= val_step
            validation_epoch_loss_values.append((epoch, val_loss))

            if(val_loss<best_metric):

                best_metric = val_loss
                best_metric_epoch = epoch

                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

                processing.save_ckp(checkpoint, relative_path_to_results + '/model/epoch_'+ str(epoch) +'validation_best_metric_model.pth')

                print('Saved new best metric model in validation.')



            #if(val_loss<best_metric):
            #    best_metric = val_loss
            #    best_metric_epoch = epoch
            #    torch.save(model.state_dict(), relative_path_to_results + '/model/epoch_'+ str(epoch) +'_best_metric_model.pth')
            #    print('Saved new best metric model')
                    

            print(f"Summary | Validation | Epoch {epoch + 1}/{max_epochs} | Average loss {val_loss:.4f}")
            print(f"Current metric {val_loss:.4f} | Best metric {best_metric:.4f} (Epoch {best_metric_epoch + 1}/{max_epochs})")
            print("All metrics and images saved")

    #if (epoch == 0):
    #    torch.save(model.state_dict(), relative_path_to_results + '/model/epoch_'+ str(epoch) +'_best_metric_model.pth')
    #    print('Saved model after one epoch for testing')
    
    'Save metrics'
    with open(relative_path_to_results + '/metrics/' + 'training_epoch_loss_values.txt', 'w') as filehandle:
        json.dump(training_epoch_loss_values, filehandle)
    with open(relative_path_to_results + '/metrics/' + 'validation_epoch_loss_values.txt', 'w') as filehandle:
        json.dump(validation_epoch_loss_values, filehandle)
    with open(relative_path_to_results + '/currupted_ids.txt', 'w') as filehandle:
        json.dump(corrupted_ids, filehandle)


'Save plot of metrics'
plt.figure('train', (12, 6))
plt.subplot(1, 2, 1)
plt.title('Training - Loss')
x = [i[0] for i in training_epoch_loss_values]
y = [i[1] for i in training_epoch_loss_values]
plt.xlabel('epoch')
plt.plot(x, y, color='red')
plt.subplot(1, 2, 2)
plt.title('Validation - Loss')
x = [i[0] for i in validation_epoch_loss_values]
y = [i[1] for i in validation_epoch_loss_values]
plt.xlabel('epoch')
plt.plot(x, y, color='green')
plt.savefig(relative_path_to_results + '/plots/' + 'training__and_validation.png')

        
#dataset.__getitem__(0)
