import cv2
import numpy as np


def get_leava_area_and_contour(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshed_img = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)

    # 去噪
    _, contours, hierarchy = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_areas = [i for i in contours if cv2.contourArea(i) < 500]
    cv2.fillPoly(threshed_img, small_areas, 255)
    threshed_img = 255 - threshed_img
    _, contours, hierarchy = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_areas = [i for i in contours if cv2.contourArea(i) < 500]
    cv2.fillPoly(threshed_img, small_areas, 0)
    _, contours, hierarchy = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_holes_area = [i for i in contours if cv2.contourArea(i) < 500]
    cv2.fillPoly(threshed_img, small_holes_area, 255)
    _, contours, hierarchy = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    small_contours_idx = [i for i in range(len(contours)) if len(contours[i]) < 50]
    for i in small_contours_idx[::-1]:
        contours.pop(i)

    for i in range(len(contours)):
        t = []
        for j in contours[i].tolist():
            t.append(j[0])
        contours[i] = np.array(t)
    cnt = np.array(contours)
    leave_area = cv2.cvtColor(threshed_img, cv2.COLOR_GRAY2RGB)

    return leave_area, cnt
