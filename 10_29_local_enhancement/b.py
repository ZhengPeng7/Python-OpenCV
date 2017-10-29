import cv2
import numpy as np
import matplotlib.pyplot as plt
import correct_angle_by_radon


image = r'./0.jpg'

img_rotated, rotation_angle = correct_angle_by_radon.correct_angle_by_radon(image)

x1 = cv2.Sobel(img_rotated, cv2.CV_16S, 1, 0, ksize=1)
x3 = cv2.Sobel(img_rotated, cv2.CV_16S, 1, 0, ksize=3)
x5 = cv2.Sobel(img_rotated, cv2.CV_16S, 1, 0, ksize=5)
x7 = cv2.Sobel(img_rotated, cv2.CV_16S, 1, 0, ksize=7)

absX1 = cv2.convertScaleAbs(x1)  # 转回uint8
absX3 = cv2.convertScaleAbs(x3)
absX5 = cv2.convertScaleAbs(x5)
absX7 = cv2.convertScaleAbs(x7)

# dst = cv2.addWeighted(absX, 0.9, absY, 0.1, 0)

cv2.imshow("absX1", absX1)
cv2.imshow("absX3", absX3)
cv2.imshow("absX5", absX5)
cv2.imshow("absX7", absX7)
print(absX1.shape)
absX1 = cv2.cvtColor(absX1, cv2.COLOR_BGR2GRAY)

# Threshold
X1_thresholded = cv2.adaptiveThreshold(absX1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 7)
cv2.imshow("X1_thresholded", X1_thresholded)

# cv2.imshow("Result", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()