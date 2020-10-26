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


'Print method evokation'
print('-' * 60)
print('Started pipeline')
print('Cuda?: ' + str(torch.cuda.is_available()))
if(torch.cuda.is_available() == True):
    device = torch.device('cuda:0')

'Parameters'
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
training_data = dataset.Dataset(all_ids, relative_path_to_scenes)
training_generator = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=True)

'Create model, loss function and optimizer'
model = large_net.MPIPredictionNet()
if(torch.cuda.is_available() == True):
    model.to(device)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=2e-4, amsgrad=True)

'Loss for training and validation'
training_epoch_loss_values = list()

'Start training process'
print('-' * 60)
print('Start training:')
for epoch in range(max_epochs):
    print('-' * 10)
    print(f"Epoch {epoch + 1}/{max_epochs}")
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
        print(f"{step}/{len(all_ids) // training_generator.batch_size}, train_loss: {loss.item():.4f}")
    
    epoch_loss /= step
    training_epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
#dataset.__getitem__(0)
