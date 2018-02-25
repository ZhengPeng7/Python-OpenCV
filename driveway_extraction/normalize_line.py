import cv2
import numpy as np
from skimage.morphology import skeletonize


def normalize_line(ran=list(range(1, 5))):
    for idx in range(1, 5):
        image = "flooded_"+str(idx)+".jpg"
        road = cv2.imread(image)
        road_gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
        ret, thr = cv2.threshold(road_gray, 10, 255, cv2.THRESH_BINARY)
        _, cnts, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        small_cnts = [i for i in cnts if cv2.contourArea(i) < 700]
        _ = cv2.fillPoly(thr, small_cnts, 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(thr, kernel)
        dilated = cv2.dilate(dilated, kernel)
        dilated = (skeletonize(cv2.threshold(dilated, 20, 255, cv2.THRESH_BINARY)[1]//255)*255).astype(np.uint8)
        dilated = cv2.dilate(dilated, kernel)
        cv2.imwrite("flooded_"+str(idx)+".jpg", dilated)


if __name__ == '__main__':
    normalize_line()