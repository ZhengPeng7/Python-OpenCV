import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def zmMinFilterGray(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    if r <= 0:
        return src
    h, w = src.shape[:2]
    I = src
    res = np.minimum(I, I[[0] + list(range(h - 1)), :])
    res = np.minimum(res, I[list(range(1, h)) + [h - 1], :])
    I = res
    res = np.minimum(I, I[:, [0] + list(range(w - 1))])
    res = np.minimum(res, I[:, list(range(1, w)) + [w - 1]])
    return zmMinFilterGray(res, r - 1)


def guidedfilter(I, p, r, eps):
    '''引导滤波，直接参考网上的matlab代码'''
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def getV1(m, r, eps, w, maxV1):  # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)  # 得到暗通道图像
    V1 = guidedfilter(V1, zmMinFilterGray(V1, 7), r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()

    V1 = np.minimum(V1 * w, maxV1)  # 对值范围进行限制

    return V1, A


def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    V1, A = getV1(m, r, eps, w, maxV1)  # 得到遮罩图像和大气光照
    for k in range(3):
        Y[:, :, k] = (m[:, :, k] - V1) / (1 - V1 / A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作
    return Y


def draw_way(image):
    m = deHaze(cv2.imread(image) / 255.0) * 255
    img = cv2.imread(image)
    cv2.imwrite("dehazed.jpg", m)
    img_dehazed = cv2.imread("dehazed.jpg")
    gray_dehazed = cv2.cvtColor(img_dehazed, cv2.COLOR_BGR2GRAY)
    # ret, thr = cv2.threshold(gray_dehazed, 120, 255, cv2.THRESH_BINARY)
    thr = cv2.Canny(gray_dehazed, 60, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thr = cv2.dilate(thr, kernel)
    # plt.imshow(thr, cmap="gray")
    # plt.show()
    # filter the contours with large area or small perimeter
    _, cnts, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    large_area_cnts = [i for i in cnts if cv2.contourArea(i) > (thr.shape[0] * thr.shape[1]) / 30]
    cv2.fillPoly(thr, large_area_cnts, 0)
    # plt.imshow(thr, cmap="gray")
    # plt.title("fill bigs")
    # plt.show()
    _, cnts, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_peri_cnts = [i for i in cnts
                       if cv2.arcLength(i, closed=True) < np.max([cv2.arcLength(t, closed=True) for t in cnts]) / 3]
    cv2.fillPoly(thr, small_peri_cnts, 0)
    _, cnts, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # squeezed_cnts = [i for i in cnts
    #                  if np.sqrt((np.max(np.squeeze(i, axis=1)[:, 0]) -
    #                              np.min(np.squeeze(i, axis=1)[:, 0])) ** 2 +
    #                             (np.max(np.squeeze(i, axis=1)[:, 1]) -
    #                              np.min(np.squeeze(i, axis=1)[:, 1])) ** 2) >
    #                  cv2.arcLength(i, closed=True) * 3]
    squeezed_cnts = [i for i in cnts if np.max(np.squeeze(i, axis=1)[:, 1]) -
                     np.min(np.squeeze(i, axis=1)[:, 1]) < thr.shape[0] / 4]
    cv2.fillPoly(thr, squeezed_cnts, 0)
    plt.imshow(thr, cmap="gray")
    plt.title("fill squeezed ones")
    plt.show()
    cnts_valid = [i.tolist() for i in cnts]
    for i in large_area_cnts:
        if i.tolist() in cnts_valid:
            cnts_valid.remove(i.tolist())
    for i in small_peri_cnts:
        if i.tolist() in cnts_valid:
            cnts_valid.remove(i.tolist())
    for i in squeezed_cnts:
        if i.tolist() in cnts_valid:
            cnts_valid.remove(i.tolist())
    cnts_valid = np.array([np.array(i) for i in cnts_valid])
    print("cnts_valid:", len(cnts_valid))
    for i in range(len(cnts_valid)):
        cv2.drawContours(img, cnts_valid, i, (0, 0, 255), thickness=5)
    cv2.imwrite("thr_" + image.split("\\")[-1], thr)
    cv2.imwrite("canny_" + image.split("\\")[-1], img)
    plt.imshow(thr, cmap="gray")
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


# 一个重要的判定标准--正常马路线的范围肯定要是横跨或纵跨近半张图片的, 
# 所以: 可借此删去面积恰当且周长恰当的区域.
if __name__ == '__main__':
    images = [os.path.join('./six_ways', i) for i in os.listdir("./six_ways")]
    images.sort()
    for image in [images[1]]:
        print(image)
        draw_way(image)
        # lines = cv2.HoughLines(thr, 1, np.pi/180, 200)
        # img_houghed = cv2.imread("dehazed.jpg")
        # print("lines[0]:", lines[0])
        # for i in range(min(10, lines.shape[0])):
        #     for rho, theta in lines[i]:
        #         a = np.cos(theta)
        #         b = np.sin(theta)
        #         x0 = a*rho
        #         y0 = b*rho
        #         x1 = int(x0 + 1000*(-b))
        #         y1 = int(y0 + 1000*(a))
        #         x2 = int(x0 - 1000*(-b))
        #         y2 = int(y0 - 1000*(a))

        #     cv2.line(img_houghed, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # cv2.imwrite('houghlines3.jpg', img_houghed)
