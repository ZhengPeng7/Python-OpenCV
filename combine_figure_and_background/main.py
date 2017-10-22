import cv2
import numpy as np


# get captures
cap = cv2.VideoCapture(0)
background_capture = cv2.VideoCapture(r'./a.avi')

while cap.isOpened():

    # extract your figure
    # refered from https://docs.opencv.org/trunk/d8/d83/tutorial_py_grabcut.html
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask = np.zeros(frame.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, 450, 490)
    cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), (0,), (1,)).astype('uint8')
    frame = frame * mask2[:, :, np.newaxis]

    # extract the background
    ret, background = background_capture.read()
    background = cv2.resize(background, (640, 480), interpolation=cv2.INTER_AREA)
    # maybe the default size of embedded camera is 640x480

    # combine the figure and background using mask instead of iteration
    mask_1 = frame > 0
    mask_2 = frame <= 0
    combination = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) * mask_1 + background * mask_2

    cv2.imshow('combination', combination)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
