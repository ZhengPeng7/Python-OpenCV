import cv2
import leave_splite
import get_images
import extract_vein_by_region_grow
import show_images


# get images
leave_splite.split_save_images(r'./split_before/', r'./split_after/')
images = get_images.get_images(r'./split_after')


# parameters settings
canny_threshold = [60, 90]
threshold_perimeter = 50
threshold_kernel_boundary = (9, 9)

vein_joined = []
images_shape = []
# extracting
for i in images:
    vein_joined.append(extract_vein_by_region_grow.extract_vein_by_region_grow(i,
                                                                               canny_threshold,
                                                                               threshold_perimeter,
                                                                               threshold_kernel_boundary))
    images_shape.append(cv2.imread(i).shape[:2])

column = 5
show_images.show_images(vein_joined, images_shape, column, alignment='left')
