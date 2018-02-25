import cv2
import numpy as np


def clockwise_or_anticlockwise(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    canny_threshold1 = [50, 150]
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)

    while lines is None:
        # 调整canny的阈值至合适
        print(lines)
        canny_threshold1 = [int(canny_threshold1[0]/2), int(canny_threshold1[1]/2)]
        edges = cv2.Canny(img, *canny_threshold1, apertureSize=3)
        lines = cv2.HoughLines(edges,1,np.pi/180, 118)

    theta = lines[0][0][1]
    return theta > 1.57
