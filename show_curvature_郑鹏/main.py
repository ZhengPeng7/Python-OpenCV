import numpy as np
import cv2
import copy
from skimage import measure, color
import get_curvature


# draw_circle函数
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        initial_img[:] = copy.deepcopy(img_cp[:])
        cv2.circle(initial_img, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(initial_img, 'curvature on (' + str(x) + ',' + str(y) + ')', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (111, 0, 111))
        if (x, y) in search_points:
            idx = search_points.index((x, y))
            print(curvature[idx])
            cv2.putText(initial_img, str(curvature[idx]), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (111, 0, 111))

# 保存原图及刷新所用备用图.
img = cv2.imread('seg_0.jpg')
initial_img = copy.deepcopy(img)
img = color.rgb2gray(img)
img_cp = copy.deepcopy(initial_img)

# 取轮廓点
contours = measure.find_contours(img, 0.60)
contours = [contours[i] for i in range(len(contours)) if len(contours[i]) > 100]
contours = np.ceil(contours)
points = [contours[0].flatten()[1::2], contours[0].flatten()[::2]]
search_points = list(zip(*points))
curvature = get_curvature.curvature_splines(*points)

# 显示
cv2.imshow('CURVATURE_OF_BOUNDARIES', initial_img)
cv2.namedWindow('CURVATURE_OF_BOUNDARIES')
cv2.setMouseCallback('CURVATURE_OF_BOUNDARIES', draw_circle)

while 1:
    cv2.imshow('CURVATURE_OF_BOUNDARIES', initial_img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()