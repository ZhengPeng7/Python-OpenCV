import os
import shutil
import cv2
import matplotlib.pyplot as plt
import split_leaves
import save_split_leaves
import get_images
import correct_angle_by_radon
import show_images

# clear "split_after" directory
if os.path.isdir(r'./split_after'):
    if os.listdir(r'./split_after'):
        shutil.rmtree(r'./split_after')
        os.mkdir(r'./split_after')

# get images
leave_split_before = get_images.get_images(r'./split_before')[0]
leaves_split = split_leaves.split_leaves(leave_split_before)
save_split_leaves.save_split_leaves(leaves_split, leave_split_before, r'./split_after')
images = get_images.get_images(r'./split_after')

imgs_rotated = []
imgs_shape = []

for image in images:
    img_rotated, correcting_angle = correct_angle_by_radon.correct_angle_by_radon(image)
    imgs_rotated.append(img_rotated)
    imgs_shape.append(img_rotated.shape)

column = 5
img_joined = show_images.show_images(imgs_rotated, imgs_shape, column, alignment='left')
img_ori = cv2.imread(leave_split_before)

# By contrast
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
ax1, ax2 = axes.ravel()
ax1.imshow(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Leaves')
ax2.imshow(img_joined)
ax2.set_title('Leaves After Correction')

plt.show()
