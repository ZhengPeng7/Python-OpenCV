# -*- coding:utf - 8 -*-
import cv2


img = cv2.imread("seg_2.jpg", 0)


def select_canny_threshold(*args):
    low = cv2.getTrackbarPos("low", "panel")
    high = cv2.getTrackbarPos("high", "panel")
    img_GB = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(img_GB, low, high)
    print(edges)
    cv2.imshow("panel", edges)

cv2.namedWindow("panel")
cv2.createTrackbar("low", "s", 0, 255, select_canny_threshold)
cv2.createTrackbar("high", "s", 0, 255, select_canny_threshold)

while True:
    select_canny_threshold()
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
