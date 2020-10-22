import dataset_processing
import dataset
import torch 

relative_path_to_scenes = 'less_data'
max_epochs = 10

all_scenes = list()
all_scenes.append('0cC7GPRFAIvP5i')
all_scenes.append('1eTVjMYXkOBq6b')
all_scenes.append('1eTVjMYXkOBq6b')
num_scenes = len(all_scenes)

all_ids = dataset_processing.generate_all_ids(all_scenes)

training_data = dataset.Dataset(all_ids, relative_path_to_scenes)
training_generator = torch.utils.data.DataLoader(training_data, batch_size=1)

for epoch in range(max_epochs):
    for data in training_generator:
        print(data['sample_id'])
#dataset.__getitem__(0)
