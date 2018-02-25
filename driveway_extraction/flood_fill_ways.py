import cv2
import numpy as np
import matplotlib.pyplot as plt
import dehaze


if __name__ == '__main__':
    import sys
    image = "./six_ways/6_point_16B_0315_49.jpg"
    img = dehaze.deHaze(cv2.imread(image) / 255.0) * 255
    cv2.imwrite("t.jpg", img)
    img = cv2.imread(image)
    h, w = img.shape[:2]  # 得到图像的高和宽
    mask = np.zeros((h + 2, w + 2), np.uint8)  # 掩码单通道8比特，长和宽都比输入图像多两个像素点，满水填充不会超出掩码的非零边缘
    seed_pt = None
    fixed_range = True
    connectivity = 4
    cv2.namedWindow("floodfill", 0)


    def update(dummy=None, filler_color=(0, 0, 255)):
        if seed_pt is None:
            cv2.imshow('floodfill', img)
            return
        flooded = img.copy()  # 以副本的形式进行填充，这样每次
        mask[:] = 0  # 掩码初始为全0
        lo = cv2.getTrackbarPos('lo', 'floodfill')  # 观察点像素邻域负差最大值（也就是与选定像素多少差值内的归为同一区域）
        hi = cv2.getTrackbarPos('hi', 'floodfill')  # 观察点像素邻域正差最大值
        flags = connectivity  # 低位比特包含连通值, 4 (缺省) 或 8
        if fixed_range:
            flags |= cv2.FLOODFILL_FIXED_RANGE  # 考虑当前象素与种子象素之间的差（高比特也可以为0）
        # 以白色进行漫水填充
        cv2.floodFill(flooded, mask, seed_pt, filler_color,
                      (lo,) * 3, (hi,) * 3, flags)

        cv2.circle(flooded, seed_pt, 2, (0, 0, 255), -1)  # 选定基准点用白色圆点标出
        line = flooded.copy()
        line[np.where(flooded != (0, 0, 255))] = 0
        # line = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
        # ret, line = cv2.threshold(line, 10, 255, cv2.THRESH_BINARY)
        # _, cnts, _ = cv2.findContours(line, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # small_area_cnts = [i for i in cnts
        #                    if cv2.arcLength(i, closed=True) < np.max([cv2.arcLength(t, closed=True) for t in cnts]) / 3]
        # cv2.fillPoly(line, small_area_cnts, 0)
        cv2.imwrite("./flooded.jpg", line)
        cv2.imshow('floodfill', line)


    def onmouse(event, x, y, flags, param):  # 鼠标响应函数
        global seed_pt
        if flags & cv2.EVENT_FLAG_LBUTTON:  # 鼠标左键响应，选择漫水填充基准点
            seed_pt = x, y
            update()


    update()
    cv2.setMouseCallback('floodfill', onmouse)
    cv2.createTrackbar('lo', 'floodfill', 20, 255, update)
    cv2.createTrackbar('hi', 'floodfill', 20, 255, update)

    while True:
        ch = 0xFF & cv2.waitKey()
        if ch == 27:
            break
        if ch == ord('f'):
            fixed_range = not fixed_range  # 选定时flags的高位比特位0，也就是邻域的选定为当前像素与相邻像素的的差，这样的效果就是联通区域会很大
            print('using %s range' % ('floating', 'fixed')[fixed_range])
            update()
        if ch == ord('c'):
            connectivity = 12 - connectivity  # 选择4方向或则8方向种子扩散
            print('connectivity =', connectivity)
            update()
    cv2.destroyAllWindows()