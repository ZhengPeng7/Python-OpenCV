import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


idx = 4
image = "flooded_"+str(idx)+".jpg"
road = cv2.imread(image)
road_gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
ret, thr = cv2.threshold(road_gray, 10, 255, cv2.THRESH_BINARY)
_, cnts, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
small_cnts = [i for i in cnts if cv2.contourArea(i) < 10]
_ = cv2.fillPoly(road_gray, small_cnts, 0)
_ = cv2.fillPoly(thr, small_cnts, 0)
plt.imshow(thr, cmap="gray")
plt.title(image)
plt.show()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilated = cv2.dilate(thr, kernel)
dilated = cv2.dilate(dilated, kernel)
dilated = (skeletonize(cv2.threshold(dilated, 20, 255, cv2.THRESH_BINARY)[1]//255)*255).astype(np.uint8)
dilated = cv2.dilate(dilated, kernel)
plt.imshow(dilated, cmap="gray")
plt.show()
cv2.imwrite("flooded_"+str(idx)+".jpg", dilated)