import os
import torch
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


'Paths'
path_to_all_images = 'results_for_plotting/all'
path_to_series_net1 = 'results_for_plotting/series_net1'
path_to_series_net2 = 'results_for_plotting/series_net2'
path_to_plot = 'results_for_plotting'
path_to_trainloss_approach1 = 'results_approach1/metrics/training_epoch_loss_values.txt'
path_to_valloss_approach1 = 'results_approach1/metrics/validation_epoch_loss_values.txt'

'''
'Plot images'
scenes = os.listdir(path_to_all_images)

fig = plt.figure(figsize=(50,60))

w, h = 5, 6
image_list = [[0 for x in range(w)] for y in range(h)] 

i = 0
j = 0

scenes.sort()
for scene in scenes: 
    if(i==h):
        i = 0 
        j += 1
    image_list[i][j] = plt.imread(os.path.join(path_to_all_images, scene))
    i+=1

titles = ['GT', 'net1 ep=1', 'net1 ep=10', 'net1 ep=20', 'net2 ep=1']

count = 0 
    
for i in range(h):
    for j in range(w):
        a = fig.add_subplot(h, w, count+1)
        a.set_title(image_list[i][0])
        imgplot = plt.imshow(image_list[i][j])
        a.set_title(titles[j])
        count += 1

plt.savefig(os.path.join(path_to_plot, 'all_plots.png'))
'''

'Plot loss'
with open(path_to_trainloss_approach1) as f:
  training_epoch_loss_values = json.load(f)
with open(path_to_valloss_approach1) as f:
  validation_epoch_loss_values = json.load(f)


plt.figure('train', (12, 6))
plt.subplot(1, 2, 1)
plt.title('Training - Loss')
x = [i[0]+1 for i in training_epoch_loss_values]
y = [i[1] for i in training_epoch_loss_values]
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.xlim(0,20)
plt.xticks(np.arange(0, 20, step=2))
plt.plot(x, y, color='red')
plt.subplot(1, 2, 2)
plt.title('Validation - Loss')
x = [i[0]+1 for i in validation_epoch_loss_values]
y = [i[1] for i in validation_epoch_loss_values]
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.xlim(0,20)
plt.xticks(np.arange(0, 20, step=2))
plt.plot(x, y, color='green')
plt.savefig(os.path.join(path_to_plot, 'training_and_validation.png'))
