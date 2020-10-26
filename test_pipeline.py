import dataset_processing
import dataset
import torch 
import processing
import get_data
import small_net
import large_net
import matplotlib.pyplot as plt
import numpy as np
import pipeline
from torch.utils.data.sampler import SubsetRandomSampler


'Print method evokation'
print('-' * 60)
print('Started pipeline')
print('Cuda?: ' + str(torch.cuda.is_available()))
if(torch.cuda.is_available() == True):
    device = torch.device('cuda:0')

'Parameters'
relative_path_to_model = 'model'
relative_path_to_scenes = 'less_data'
max_epochs = 10

'Example dataset'
all_scenes = list()
all_scenes.append('0cC7GPRFAIvP5i')
all_scenes.append('1eTVjMYXkOBq6b')
all_scenes.append('1eTVjMYXkOBq6b')
num_scenes = len(all_scenes)
all_ids = dataset_processing.generate_all_ids(all_scenes)

'Create customized pytorch dataset'
validation_split = .2
random_seed = 42
all_data = dataset.Dataset(all_ids, relative_path_to_scenes)
all_data_size = len(all_data)
indices = list(range(all_data_size))
split = int(np.floor(validation_split * all_data_size))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

training_generator = torch.utils.data.DataLoader(all_data, batch_size=1, sampler=valid_sampler)
validation_generator = torch.utils.data.DataLoader(all_data, batch_size=1, sampler=valid_sampler)


'Create model, loss function and optimizer'
model = large_net.MPIPredictionNet()
if(torch.cuda.is_available() == True):
    model.to(device)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=2e-4, amsgrad=True)

'Loss for training and validation'
training_epoch_loss_values = list()
validation_epoch_loss_values = list()
best_metric = 0
best_metric_epoch = 0

'Start training process'
print('-' * 60)
print('Start training:')
for epoch in range(max_epochs):
    print('-' * 10)
    print(f"Training - Epoch {epoch + 1}/{max_epochs}")
    epoch_loss = 0
    step = 0
    model.train()
    if(torch.cuda.is_available() == True):
        torch.cuda.empty_cache()

    'Forward and backward in batches (batches with 2x5 PSVs are assembled inside dataloader, batch_size therefore 1'
    for data in training_generator:
        step += 1
        print('Sample: ' + data['sample_id'][0])
        optimizer.zero_grad()
        psvs, target_image = torch.squeeze(data['psvs']), torch.squeeze(data['target_image'])
        if(torch.cuda.is_available() == True):
            psvs.to(device)
            target_image.to(device)
        mpis = model(psvs)
        loss = loss_function(target_image, pipeline.get_target_image(mpis))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_indices) // training_generator.batch_size}, train_loss: {loss.item():.4f}")
    
    epoch_loss /= step
    training_epoch_loss_values.append((epoch, epoch_loss))
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")


    'Validate for every 10th epoch'
    val_interval = 10
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            print('-' * 60)
            print('Evaluation - epoch {}'.format(epoch+1))
            val_loss = 0
            val_step = 0
            for data in validation_generator:
                val_step += 1
                psvs, target_image = torch.squeeze(data['psvs']), torch.squeeze(data['target_image'])
                if(torch.cuda.is_available() == True):
                    psvs.to(device)
                    target_image.to(device)
                mpis = model(psvs)
                loss = loss_function(target_image, pipeline.get_target_image(mpis))
                val_loss += loss
            val_loss /= val_step
            validation_epoch_loss_values.append((epoch, val_loss))

            if(val_loss<best_metric):
                best_metric = val_loss
                best_metric_epoch = epoch
                torch.save(model.state_dict(), relative_path_to_model + str('/'+ epoch +'_best_metric_model.pth'))
                print('Saved new best metric model')
                    
            print('Validation - Current metric: ' + str(val_loss))
            print('Validation - Best metric: ' + str(best_metric) + '[epoch: ' + best_metric_epoch + ']')
        
    if (epoch == 1):
        print('Saved model after one epoch for testing')
        torch.save(model.state_dict(), relative_path_to_model + str('/'+ epoch +'_best_metric_model.pth'))


        
#dataset.__getitem__(0)
