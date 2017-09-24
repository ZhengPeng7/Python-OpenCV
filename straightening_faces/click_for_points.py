import cv2
import numpy as np


def click_for_points(img_original):
    # container:
    axis_original = []

    # mouse callback function
    def show_axis_original(event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONUP:
            axis_original.append([y, x])

    cv2.namedWindow('img_original')
    cv2.setMouseCallback('img_original', show_axis_original)

    while len(axis_original) < 3:
        cv2.imshow('img_original', img_original)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    # print('axis_original:', axis_original)
    cv2.destroyAllWindows()

    return np.float32(axis_original)
