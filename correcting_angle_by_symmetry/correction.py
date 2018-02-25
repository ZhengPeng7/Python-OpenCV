import cv2
import matplotlib.pyplot as plt
import get_pic_rotated_and_broaden
import clockwise_or_anticlockwise


def correction(image):
    """
        对具有对称性的图像进行旋转扶正,
        返回扶正后的图像和纠正角
    """
    img = cv2.imread(image)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bw = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
    # 填充大块中的空白, 去掉零散的小块, 使判断更精确
    image, contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_areas = [i for i in contours if cv2.contourArea(i) < 200]
    big_areas = [i for i in contours if cv2.contourArea(i) > 2000]
    cv2.fillPoly(bw, small_areas, 0)
    cv2.fillPoly(bw, big_areas, 255)

    rotation_angle = 0
    rotated_pic = get_pic_rotated_and_broaden.get_pic_rotated_and_broaden(bw, rotation_angle)
    show_counter = 1
    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    ax0, ax1 = axes.ravel()
    ax0.imshow(rotated_pic, plt.cm.gray)
    ax0.set_title('original_img')
    symmetry_rate = 0       # 对称率
    extra_search_time = 3   # 对称率不必上一次高的情况下, 额外探出的步数
    correction_angle = 0    # 纠正角
    cw_or_acw = clockwise_or_anticlockwise.clockwise_or_anticlockwise(img)  # 根据最长线判断顺时针还是逆时针转
    while True:
        show_counter += 1
        rotation_angle += (2 * ((-1) ** cw_or_acw))     # 每次旋转角度(步长)为2度
        rotated_pic = get_pic_rotated_and_broaden.get_pic_rotated_and_broaden(bw, rotation_angle)
        [_, w] = rotated_pic.shape
        # 统计符合对称的像素点数
        t_and_array = cv2.bitwise_and(rotated_pic[:, :int(w/2)], rotated_pic[:, 2*int(w/2)-1:int(w/2)-1:-1])
        t = 0
        for i in t_and_array:
            for j in i:
                t += int(j/255)
        if symmetry_rate < t:   # 比较, 找到最符合的
            symmetry_rate = t
            correction_angle = rotation_angle
            extra_search_time = 3
        else:
            extra_search_time -= 1
        if extra_search_time < 1:
            final_pic = get_pic_rotated_and_broaden.get_pic_rotated_and_broaden(bw, correction_angle)
            ax1.imshow(final_pic, plt.cm.gray)
            ax1.set_title('final_img')
            break

    plt.show()
    return final_pic, correction_angle
