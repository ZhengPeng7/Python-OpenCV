from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
import get_leave_area_and_contour
import get_pic_rotated_and_broaden


def correcting_angle_by_hull_slope(image):
    img = cv2.imread(image)
    # img = img[:-100, :]     # 防止边缘有拍出白纸的区域

    # 提取叶片
    leave_area, cnt = get_leave_area_and_contour.get_leava_area_and_contour(img)
    # cv2.drawContours(leave_area, cnt, 0, (255, 0, 0))
    hull = cv2.convexHull(cnt)
    hull = np.array([np.squeeze(hull).tolist()])

    cv2.drawContours(leave_area, hull, 0, (0, 0, 255), thickness=2)

    # hull[0][:, 0], hull[0][:, 1] = hull[0][:, 1], hull[:, 0]
    t = hull[0][:, 0].copy()
    hull[0][:, 0] = hull[0][:, 1].copy()
    hull[0][:, 1] = t
    # print('hull:', hull)
    plt.figure(1)
    plt.ion()
    plt.imshow(leave_area)
    for i in range(len(hull[0])):
        if np.sqrt((hull[0][i % len(hull[0])][0] - hull[0][(i+1) % len(hull[0])][0]) ** 2 +
                   (hull[0][i % len(hull[0])][1] - hull[0][(i+1) % len(hull[0])][1]) ** 2) < 2\
                or not (hull[0][i % len(hull[0])][0] - hull[0][(i+1) % len(hull[0])][0]):
            hull[0][(i+1) % len(hull[0])][0], hull[0][(i+1) % len(hull[0])][1] = \
                round((hull[0][(i % len(hull[0]))][0]+hull[0][((i+2) % len(hull[0]))][0]) / 2), \
                round((hull[0][(i % len(hull[0]))][1]+hull[0][((i+2) % len(hull[0]))][1]) / 2)
            continue
        plt.scatter(hull[0][i][1], hull[0][i][0])
        plt.show()
        plt.pause(0.01)
    plt.ioff()

    tangents = []
    weights = []
    # angles = []
    for i in range(len(hull[0]) - 1):
        # if hull[0][i][1] - hull[0][i+1][1] < 1 or current_weight < 5:
        #     hull[0][i+1][0], hull[0][i+1][1] = round(sum(hull[0][i:i+2][0])/2), round(sum(hull[0][i:i+2][1])/2)
        #     continue
        # else:
        current_tangent = 1 / ((hull[0][i][0] - hull[0][i+1][0]) / (hull[0][i][1] - hull[0][i+1][1]))     # 问题可能出在此处.
        current_weight = np.sqrt((hull[0][i][0] - hull[0][i+1][0]) ** 2 + (hull[0][i][1] - hull[0][i+1][1]) ** 2)
        # if current_tangent > 10000:       # threshold=100
        #     continue
        tangents.append(current_tangent)
        weights.append(current_weight)
        # print('{}th: current_tangent={}, current_weight={}'.format(i, current_tangent, current_weight))
    tangents = np.array(tangents)
    weights_normalized = np.array(weights) / sum(weights)
    tangents_weighted = np.array(tangents) * weights_normalized
    correction_angle = np.arctan(sum(tangents_weighted)) * 180 / np.pi

    # print('tangents:', tangents)
    # print('weights:', weights)
    # print('weights_normalized:', weights_normalized)
    print('tangents_weighted:', tangents_weighted)
    print('correction_angle:', correction_angle)
    # print('hull:', hull)


    img_rotated = get_pic_rotated_and_broaden.get_pic_rotated_and_broaden(img, -correction_angle)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax0, ax1 = axes.ravel()
    ax0.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax0.set_title('Ori')
    ax1.imshow(cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB))
    ax1.set_title('Rot')
    plt.show()
