import cv2
import numpy as np
import region_grow
import get_boundary


def extract_vein_by_region_grow(image, canny_threshold, threshold_perimeter, threshold_kernel_boundary):
    """
    function: 通过区域增长法提取叶脉
    :param image: 源图片路径
    :param canny_threshold: Canny算子的上下限
    :param threshold_perimeter: 按区域周长除噪的长度阈值
    :param threshold_kernel_boundary: 边界膨胀的kernel的阈值
    :return: vein叶脉图
    """

    img = cv2.imread(image)

    # Canny
    img_GB = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(img_GB, *canny_threshold, apertureSize=3)

    # 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.dilate(edges, kernel)
    opened = cv2.dilate(opened, kernel)
    opened = cv2.dilate(opened, kernel)
    opened = cv2.erode(opened, kernel)
    opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, kernel2)

    # 依面积去噪
    opened, contours, hierarchy = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_areas = [i for i in contours if cv2.contourArea(i) < 200]
    big_areas = [i for i in contours if cv2.contourArea(i) > 2000]
    cv2.fillPoly(opened, small_areas, 0)
    cv2.fillPoly(opened, big_areas, 255)

    # 某方位区域增长
    res_all = region_grow.region_grow(opened, 'all')

    # get boundary
    boundary = get_boundary.get_boundary(image)
    canvas_boundary = np.zeros(img.shape[:2], dtype=np.uint8)
    for i in boundary:
        canvas_boundary[int(i[0]), int(i[1])] = 255
    kernel_boundary = cv2.getStructuringElement(cv2.MORPH_RECT, threshold_kernel_boundary)
    canvas_boundary = cv2.dilate(canvas_boundary, kernel_boundary)

    # 得到叶脉并依区域周长去噪
    vein = cv2.subtract(res_all, canvas_boundary)
    vein, contours, hierarchy = cv2.findContours(vein, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_perimeters = [i for i in contours if len(i) < threshold_perimeter]   # 删短周长的区域
    cv2.fillPoly(vein, small_perimeters, 0)
    vein, contours, hierarchy = cv2.findContours(vein, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return vein
