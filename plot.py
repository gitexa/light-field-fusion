import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



'Images'
i1_gt = plt.imread('images/0x8IbwzW484CWq_13_17_target_pose_15_epoch9_target_image.png')
i1_ap_1 = plt.imread('images/0x8IbwzW484CWq_13_17_target_pose_15_epoch_9_predicted_image_loss_0.0046305167488753796.png')
i1_ap_2 = plt.imread('images/0x8IbwzW484CWq_13_17_target_pose_15_epoch_9_predicted_image_loss_0.0046305167488753796.png')
i1_ap_3 = plt.imread('images/0x8IbwzW484CWq_13_17_target_pose_15_epoch_9_predicted_image_loss_0.0046305167488753796.png')
i2_gt = plt.imread('images/a1dUOWBAZyuzkO_15_55_target_pose_35_epoch9_target_image.png')
i2_ap_1 = plt.imread('images/a1dUOWBAZyuzkO_15_55_target_pose_35_epoch_9_predicted_image_loss_0.0015275815967470407.png')
i2_ap_2 = plt.imread('images/a1dUOWBAZyuzkO_15_55_target_pose_35_epoch_9_predicted_image_loss_0.0015275815967470407.png')
i2_ap_3 = plt.imread('images/a1dUOWBAZyuzkO_15_55_target_pose_35_epoch_9_predicted_image_loss_0.0015275815967470407.png')
i3_gt = plt.imread('images/ApjUMjuSV66Jg9_26_22_target_pose_24_epoch9_target_image.png')
i3_ap_1 = plt.imread('images/ApjUMjuSV66Jg9_26_22_target_pose_24_epoch_9_predicted_image_loss_0.0015174155123531818.png')
i3_ap_2 = plt.imread('images/ApjUMjuSV66Jg9_26_22_target_pose_24_epoch_9_predicted_image_loss_0.0015174155123531818.png')
i3_ap_3 = plt.imread('images/ApjUMjuSV66Jg9_26_22_target_pose_24_epoch_9_predicted_image_loss_0.0015174155123531818.png')

title_gt = 'GT'
title_ap_1 = 'Approach 1'
title_ap_2 = 'Approach 2'
title_ap_3 = 'Approach 3'

imgs = list()
imgs.append((title_gt, i1_gt))
imgs.append((title_ap_1, i1_ap_1))
imgs.append((title_ap_2, i1_ap_2))
imgs.append((title_ap_3, i1_ap_3))
imgs.append((title_gt, i2_gt))
imgs.append((title_ap_1, i2_ap_1))
imgs.append((title_ap_2, i2_ap_2))
imgs.append((title_ap_3, i2_ap_3))
imgs.append((title_gt, i3_gt))
imgs.append((title_ap_1, i3_ap_1))
imgs.append((title_ap_2, i3_ap_2))
imgs.append((title_ap_3, i3_ap_3))

fig = plt.figure()

for i in range(12):
    a = fig.add_subplot(3, 4, i+1)
    imgplot = plt.imshow(imgs[i][1])
    a.set_title(imgs[i][0])
plt.show()