import cv2
import matplotlib.pyplot as plt
import get_images
import local_enhancement
import os
import shutil
import split_leaves
import save_split_leaves
import correct_angle_by_radon
import show_images
import extract_vein_by_region_grow

# # clear "split_after" directory
# if os.path.isdir(r'./split_after'):
#     if os.listdir(r'./split_after'):
#         shutil.rmtree(r'./split_after')
#         os.mkdir(r'./split_after')
#
# # clear "corrected_after" directory
# if os.path.isdir(r'./corrected_after'):
#     if os.listdir(r'./corrected_after'):
#         shutil.rmtree(r'./corrected_after')
#         os.mkdir(r'./corrected_after')
#
# # get images
# leave_split_before = get_images.get_images(r'./split_before')[0]
# leaves_split = split_leaves.split_leaves(leave_split_before)
# save_split_leaves.save_split_leaves(leaves_split, leave_split_before, r'./split_after')
# images = get_images.get_images(r'./split_after')
#
# imgs_rotated = []
# imgs_shape = []
#
# # for i in get_images.get_images(r'./corrected_after'):
# #     imgs_rotated.append(cv2.imread(i))
# #     imgs_shape.append(cv2.imread(i).shape)
# for image in images:
#     img_rotated, correcting_angle = correct_angle_by_radon.correct_angle_by_radon(image)
#     cv2.imwrite(r'./corrected_after/'+image.rpartition('/')[-1].rpartition('.')[-3][-1]+'.jpg', img_rotated)
#     imgs_rotated.append(img_rotated)
#     imgs_shape.append(img_rotated.shape)
#
# column = 5
# img_joined = show_images.show_images(imgs_rotated, imgs_shape, column, alignment='left')
# img_ori = cv2.imread(leave_split_before)

# # By contrast
# fig, axes = plt.subplots(1, 2, figsize=(16, 8))
# ax1, ax2 = axes.ravel()
# ax1.imshow(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
# ax1.set_title('Original Leaves')
# ax2.imshow(img_joined)
# ax2.set_title('Leaves After Correction')

# plt.show()

images = get_images.get_images(r'./corrected_after/')

edges_canny = []
edges_equalized = []
edges_canny_shape = []
edges_equalized_shape = []

for image in images:
    img, img_equalized, edge_canny, edge_equalized = local_enhancement.local_enhancement(image)
    edges_canny.append(edge_canny)
    edges_equalized.append(edge_equalized)
    edges_canny_shape.append(edge_canny.shape)
    edges_equalized_shape.append(edge_equalized.shape)
    # plt.imshow(edge_canny, plt.cm.gray)
    # plt.suptitle('ori_edge')
    # plt.show()
    #
    # plt.subplot(121), plt.imshow(img, 'gray')
    # plt.subplot(122), plt.imshow(img_equalized, 'gray')
    # plt.show()
    #
    # plt.subplot(121), plt.imshow(edge_canny, 'gray')
    # plt.subplot(122), plt.imshow(edge_equalized, 'gray')
    # plt.show()

column = 5
edge_canny_joined = show_images.show_images(edges_canny, edges_canny_shape, column, alignment='left')
edge_equalized_joined = show_images.show_images(edges_equalized, edges_equalized_shape, column, alignment='left')

# By contrast
fig_1, axes_1 = plt.subplots(1, 1, figsize=(16, 8))
axes_1.imshow(edge_canny_joined, plt.cm.gray)
axes_1.set_title('Cannied Edges')

# Extract main vein
main_veins = []
veins = []
main_veins_points = []
veins_points = []
main_veins_shape = []
veins_shape = []
for i in range(len(edges_canny)):
    vein, main_vein, vein_points, main_vein_points = \
        extract_vein_by_region_grow.extract_vein_by_region_grow(images[i], edges_canny[i], 150, (15, 15))
    veins.append(vein)
    main_veins.append(main_vein)
    main_veins_points.append(main_vein_points)
    veins_points.append(vein_points)
    main_veins_shape.append(main_vein.shape)
    veins_shape.append(vein.shape)

column = 5
vein_joined = show_images.show_images(veins, veins_shape, column, alignment='left')
main_vein_joined = show_images.show_images(main_veins, main_veins_shape, column, alignment='left')

# By contrast
fig_2, axes_2 = plt.subplots(1, 1, figsize=(16, 8))
axes_2.imshow(vein_joined, plt.cm.gray)
axes_2.set_title('Veins')
fig_3, axes_3 = plt.subplots(1, 1, figsize=(16, 8))
axes_3.imshow(main_vein_joined, plt.cm.gray)
axes_3.set_title('Main Veins')

plt.show()
