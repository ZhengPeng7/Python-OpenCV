import cv2
import numpy as np
import region_grow
import get_boundary
import matplotlib.pyplot as plt


def extract_vein_by_region_grow(image, img, threshold_perimeter, threshold_kernel_boundary):

    # get boundary
    boundary = get_boundary.get_boundary(image)
    canvas_boundary = np.zeros(img.shape[:2], dtype=np.uint8)
    for i in boundary:
        canvas_boundary[int(i[0]), int(i[1])] = 255
    kernel_boundary = cv2.getStructuringElement(cv2.MORPH_RECT, threshold_kernel_boundary)
    canvas_boundary = cv2.dilate(canvas_boundary, kernel_boundary)  # 膨胀后的边框

    # 膨胀后的边框和原叶脉进行或操作
    opened = cv2.bitwise_or(img, canvas_boundary)

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
    # 连接断裂的主叶脉
    h, w = vein.shape
    # denoise
    vein[:, round(w / 2) - 20:round(w / 2) + 20], contours, hierarchy = \
        cv2.findContours(vein[:, round(w/2)-20:round(w/2)+20], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_perimeters = [i for i in contours if len(i) < 10]   # 删短周长的区域
    cv2.fillPoly(vein[:, round(w/2)-20:round(w/2)+20], small_perimeters, 0)
    plt.imshow(vein[:, round(w/2)-20:round(w/2)+20], plt.cm.gray)
    plt.suptitle('First One')
    plt.show()
    # temporary end
    vein_end = []
    # get the bottom index
    for end_idx in range(len(vein[::-1, round(w/2)-20:round(w/2)+20])):
        if vein[end_idx, :].any():
            for j in range(len(vein[end_idx, round(w/2)-20:round(w/2)+20])):
                if vein[:, round(w/2)-20:round(w/2)+20][end_idx][j] == 255:
                    vein_end = [end_idx, j+1]
    for i in range(len(vein[:vein_end[0], round(w/2)-20:round(w/2)+20])):
        if i != 0:
            if start_point and end_point and i in list(range(0, end_point[0])):
                continue
        print('i:{}'.format(i))
        start_point = []
        end_point = []
        flag_end = 'go'

        # get start_point and end_point
        for j in range(len(vein[i:vein_end[0], round(w/2)-20:round(w/2)+20])):
            if flag_end == 'brk':
                break
            if vein[:, round(w/2)-20:round(w/2)+20][j].any() and start_point == []:
                if not vein[:, round(w/2)-20:round(w/2)+20][j+1].any():
                    for k in range(len(vein[:, round(w/2)-20:round(w/2)+20][j])):
                        if vein[:, round(w/2)-20:round(w/2)+20][j][k] == 255:
                            start_point = [i+j, (k+round(w/2)-20)+1]
                            print('start_point:', start_point)
                            break
            elif not vein[:, round(w/2)-20:round(w/2)+20][j].any() and start_point != [] and end_point == []:
                print("All zeros in %d-th line." % j)
                if vein[:, round(w/2)-20:round(w/2)+20][j+1].any():
                    for k in range(len(vein[:, round(w/2)-20:round(w/2)+20][j])):
                        if vein[:, round(w/2)-20:round(w/2)+20][j+1][k] == 255:
                            end_point = [i+j+1, (k+round(w/2)-20)+1]
                            print('end_point:', end_point)
                            flag_end = 'brk'
                            break
            else:
                continue
        # get points end
        if not start_point or not end_point:
            continue

        canny_threshold_enhanced_locally = [30, 60]
        # print(start_point, end_point)
        # print([start_point[0], end_point[0], min(start_point[1], end_point[1]), max(start_point[1], end_point[1])])
        vein_enhanced_locally = vein[start_point[0]:end_point[0],
                                min(start_point[1], end_point[1]):max(start_point[1], end_point[1])+1]
        print('vein_enhanced_locally:')     # , vein_enhanced_locally)
        edge_enhanced_locally = cv2.Canny(vein_enhanced_locally, *canny_threshold_enhanced_locally, apertureSize=3)
        white_pixel_percentage = list(edge_enhanced_locally.ravel() == 255).count(1) / len(list(edge_enhanced_locally.ravel()))
        start_point_check = [0, 0]
        counter_canny_adjustment = 0
        counter_prevent_dead_loop = 2
        white_pixel_percentage_prev = 0
        while not 1/40 < white_pixel_percentage < 1/20 and start_point_check[1] > start_point[1]:
            if not counter_prevent_dead_loop:
                break
            # code of adjustment on threshold of canny
            if white_pixel_percentage <= 1/40:
                canny_threshold_enhanced_locally = [canny_threshold_enhanced_locally[0] - 1,
                                                    canny_threshold_enhanced_locally[1] - 1]
            else:
                canny_threshold_enhanced_locally = [canny_threshold_enhanced_locally[0] + 1,
                                                    canny_threshold_enhanced_locally[1] + 1]
            edge_enhanced_locally = cv2.Canny(vein_enhanced_locally, *canny_threshold_enhanced_locally, apertureSize=3)
            white_pixel_percentage = list(edge_enhanced_locally.ravel() == 255).count(1) / \
                                len(list(edge_enhanced_locally.ravel()))
            if white_pixel_percentage == white_pixel_percentage_prev:
                counter_prevent_dead_loop -= 1
            white_pixel_percentage_prev = white_pixel_percentage
            print('{}-th adjusted canny threshold:{}'.format(counter_canny_adjustment,
                                                           canny_threshold_enhanced_locally))
            counter_canny_adjustment += 1
            # CHECK IF THE BRANCH HAS EXTENDED
            for k in range(len(vein[:, round(w / 2) - 20:round(w / 2) + 20])):
                start_point_check = []
                if vein[:, round(w / 2) - 20:round(w / 2) + 20][k].any():
                    if not vein[:, round(w / 2) - 20:round(w / 2) + 20][k + 1].any():
                        for j in range(len(vein[:, round(w / 2) - 20:round(w / 2) + 20][k])):
                            if vein[:, round(w / 2) - 20:round(w / 2) + 20][k][j] == 255:
                                start_point_check.append([k, j + 1])
                                break
            # CHECK END
        # while end
        vein[start_point[0]:end_point[0], min(start_point[1], end_point[1]):max(start_point[1], end_point[1])+1] = \
            edge_enhanced_locally
        plt.imshow(vein, plt.cm.gray)
        plt.suptitle('In Iteration')
        plt.show()
    # for end

    print('vein:', type(vein[:, round(w/2)-20:round(w/2)+20]), vein[:, round(w/2)-20:round(w/2)+20].shape)
    # denoise
    vein[:, round(w / 2) - 20:round(w / 2) + 20], contours, hierarchy = \
        cv2.findContours(vein[:, round(w/2)-20:round(w/2)+20], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_perimeters = [i for i in contours if len(i) < 10]   # 删短周长的区域
    cv2.fillPoly(vein[:, round(w/2)-20:round(w/2)+20], small_perimeters, 0)
    plt.imshow(vein[:, round(w/2)-20:round(w/2)+20], plt.cm.gray)
    plt.suptitle('Last One')
    plt.show()

    vein, contours, hierarchy = cv2.findContours(vein, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_perimeters = [i for i in contours if len(i) < threshold_perimeter]   # 删短周长的区域
    cv2.fillPoly(vein, small_perimeters, 0)

    # 上 -> 下
    res_top = region_grow.region_grow(vein, 'top')
    kernel_artery = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    res_top = cv2.dilate(res_top, kernel_artery)
    res_top = cv2.dilate(res_top, kernel_artery)
    artery = cv2.bitwise_and(vein, res_top)
    artery, contours, hierarchy = cv2.findContours(artery, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_perimeters = [i for i in contours if len(i) < 0]   # 删短周长的区域
    cv2.fillPoly(artery, small_perimeters, 0)

    # save points
    vein_points = []
    for i in range(vein.shape[0]):
        for j in range(vein.shape[1]):
            if vein[i, j] == 255:
                vein_points.append([i, j])
    artery_points = []
    for i in range(artery.shape[0]):
        for j in range(artery.shape[1]):
            if artery[i, j] == 255:
                artery_points.append([i, j])

    # show1
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    # ax0, ax1, ax2 = axes.ravel()
    # ax0.imshow(res_all, plt.cm.gray)
    # ax0.set_title('全方位区域增长')
    # ax1.imshow(canvas_boundary, plt.cm.gray)
    # ax1.set_title('膨胀后的轮廓')
    # ax2.imshow(vein, plt.cm.gray)
    # ax2.set_title('叶脉')
    # plt.show()

    return vein, artery, vein_points, artery_points
