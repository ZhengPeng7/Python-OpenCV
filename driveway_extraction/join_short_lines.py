import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


idx = 4
image = "flooded_"+str(idx)+".jpg"
# can = np.zeros((cv2.imread("flooded_1.jpg").shape[:2]), dtype=np.uint8)
# for i in range(3):
#     image_i = image.split(".jpg")[0]+"_"+str(i)+'.jpg'
#     img = cv2.threshold(cv2.imread(image_i, 0), 20, 255, cv2.THRESH_BINARY)[1]
#     img = (skeletonize(img//255)*255).astype(np.uint8)
#     can = cv2.bitwise_or(can, img)
# can = img
img = cv2.threshold(cv2.imread(image, 0), 20, 255, cv2.THRESH_BINARY)[1]
y, x = np.where(img != 0)
f = np.polyfit(x, y, 1)
x_fit = list(range(850, 2000))
y_fit = np.polyval(f, x_fit)
print(x_fit)
for i in range(len(x_fit)):
    try:
        img[int(y_fit[i]), x_fit[i]] = 255
    except:
        pass
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilated = cv2.dilate(img.astype(np.uint8), kernel)
dilated = cv2.dilate((skeletonize(dilated//255)*255).astype(np.uint8), kernel)
plt.imshow(dilated, cmap="gray")
plt.show()
cv2.imwrite("flooded_"+str(idx)+".jpg", dilated)
