import cv2
from skimage import measure, color
import numpy as np


def get_boundary(image):
    img = cv2.imread(image)
    if len(img.shape) == 3:
        img = color.rgb2gray(img)

    contours = measure.find_contours(img, 0.68, fully_connected='high')
    contours = [contours[i] for i in range(len(contours)) if len(contours[i]) > 100]
    contours = np.array(contours)
    contours = np.squeeze(contours, axis=(0,))

    return contours
