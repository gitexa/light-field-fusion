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
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=2e-4, amsgrad=True)

'Start training process'
model.train()
for epoch in range(max_epochs):
    for data in training_generator:
        print(data['sample_id'])
        optimizer.zero_grad()
        mpis = model(torch.squeeze(data['psvs']))
        loss = loss_function(data['target_image'], pipeline.get_target_image(mpis))
        loss.backward()
        optimizer.step()
        
#dataset.__getitem__(0)
