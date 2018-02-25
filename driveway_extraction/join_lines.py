import cv2
import numpy as np
import matplotlib.pyplot as plt
import normalize_line


# normalize_line.normalize_line()
image = "./six_ways/6_point_16B_0315_49.jpg"
img = cv2.imread(image)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
can = np.zeros_like(gray, dtype=np.uint8)
for i in range(1, 5):
    can = cv2.bitwise_or(can, cv2.threshold(cv2.imread("./flooded_"+str(i)+".jpg", 0), 20, 255, cv2.THRESH_BINARY)[1])
can = cv2.cvtColor(cv2.threshold(can, 20, 255, cv2.THRESH_BINARY)[1], cv2.COLOR_GRAY2BGR)

cv2.line(can, (762, 464), (782, 501), (255, 255, 255), thickness=4)
cv2.line(can, (786, 462), (838, 495), (255, 255, 255), thickness=4)
cv2.line(can, (818, 458), (849, 472), (255, 255, 255), thickness=4)
cv2.line(can, (840, 456), (940, 499), (255, 255, 255), thickness=4)
cv2.line(can, (761, 461), (852, 461), (255, 255, 255), thickness=4)
can[1079, 1536] = 255
can[1077:, 897:900] = 255
can[:461, :] = 0
plt.imshow(cv2.cvtColor(can, cv2.COLOR_BGR2RGB))
plt.show()
areas = 255 - can
mask_ff = np.zeros((can.shape[0]+2, can.shape[1]+2), dtype=np.uint8)
cv2.floodFill(areas, mask_ff, (1, 1), 0)
cv2.floodFill(areas, mask_ff, (1750, 1000), 0)
areas = cv2.threshold(cv2.cvtColor(areas, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite("./thr_"+image.split("/")[-1][0]+".jpg", areas)
plt.imshow(areas, cmap="gray")
plt.show()
mask = cv2.threshold(cv2.cvtColor(can, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)[1] == 255
mask = cv2.cvtColor((mask*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
mask = (mask > 0).astype(np.uint8)
for i in range(can.shape[0]):
    for j in range(can.shape[1]):
        if can[i, j].all():
            can[i, j] = (0, 0, 255)
mask_ori = (1 - mask).astype(np.uint8)
road_lined = cv2.add(img*mask_ori, can*mask)
cv2.imwrite("./canny_"+image.split("/")[-1][0]+".jpg", road_lined)
plt.imshow(cv2.cvtColor(road_lined, cv2.COLOR_BGR2RGB))
plt.show()
