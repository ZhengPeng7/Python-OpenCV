import numpy as np
import cv2
import matplotlib.pyplot as plt
import get_boundary


def extract_vein_by_hough(image):
    img = cv2.imread(image)

    # 黑底画板, 用于之后存放最长线
    h, w, c = img.shape
    img_t = np.zeros((h, w), dtype=np.uint8, order='C')

    # Canny
    img = cv2.GaussianBlur(img, (3, 3), 0)
    canny_threshold1 = [50, 150]
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)
    while lines is None:
        # 调整canny的阈值至合适
        canny_threshold1 = [int(9*canny_threshold1[0]/10), int(9*canny_threshold1[1]/10)]
        edges = cv2.Canny(img, *canny_threshold1, apertureSize=3)
        lines = cv2.HoughLines(edges,1,np.pi/180, 118)
    edges_original = edges.copy()

    # 形态学操作部分
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # 边缘加粗
    opened = cv2.dilate(edges, kernel)
    opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, kernel)

    # 划线
    if lines is not None:
        for line in lines[0]:
            rho = line[0]
            theta = line[1]
            if (theta < (np.pi/4.0)) or (theta > (3.*np.pi/4.0)):
                pt1 = (int(rho/np.cos(theta)), 0)
                pt2 = (int((rho-edges.shape[0]*np.sin(theta))/np.cos(theta)), edges.shape[0])
                cv2.line(edges, pt1, pt2, 255, 20)
                cv2.line(img_t, pt1, pt2, 255, 20)
            else:
                pt1 = (0, int(rho/np.sin(theta)))
                pt2 = (edges.shape[1], int((rho-edges.shape[1]*np.cos(theta))/np.sin(theta)))
                cv2.line(edges, pt1, pt2, 255, 1)
                cv2.line(img_t, pt1, pt2, 255, 1)
    else:
        print('No line!')
        return 0
    # 与操作, 得出主叶脉
    main_vein = cv2.bitwise_and(img_t, edges_original)
    vein_and_boundary = main_vein.copy()
    boundary = get_boundary.get_boundary(image)
    for i in boundary:
        vein_and_boundary[int(i[0])][int(i[1])] = 255
    _, axes = plt.subplots(1, 3, figsize=(20, 8))
    ax0, ax1, ax2 = axes.ravel()
    ax0.imshow(edges, plt.cm.gray)
    ax0.set_title('Canny')      # 边缘检测
    ax1.imshow(opened, plt.cm.gray)
    ax1.set_title('Open')       # 线条加粗
    ax2.imshow(img_t, plt.cm.gray)
    ax2.set_title('line')       # 最长线
    _, axes = plt.subplots(1, 1, figsize=(8, 8))
    axes.imshow(vein_and_boundary, plt.cm.gray)
    axes.set_title('vein_and_boundary')
    plt.savefig(image[:image.find(r'.jpg')] + '_effect.png')
    plt.show()
    return 1
