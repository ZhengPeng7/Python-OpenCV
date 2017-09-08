import cv2
import numpy as np
import matplotlib.pyplot as plt
import get_image_broadened


def show_images(vein_joined, images_shape, column, alignment='left'):
    # 求最大长和最大宽
    h_max, w_max = np.amax(images_shape, axis=0)[:2]
    img_line_buffer = []
    img_lines = []
    counter_col = 0
    for i in range(len(vein_joined)):
        bg_color = (vein_joined[i][0][0], )
        counter_col += 1
        t = get_image_broadened.get_image_broadened(vein_joined[i], h_max, w_max)
        img_line_buffer.append(t)
        if counter_col > column - 1:
            img_lines.append(np.hstack(img_line_buffer))
            img_line_buffer = []
            counter_col = 0
            continue
        if i == len(vein_joined) - 1:
            appendix_bg_color_space = np.array([bg_color * (w_max * (column - len(vein_joined) % column))] * h_max)
            img_line_buffer = np.squeeze(img_line_buffer, axis=0)
            if alignment == 'left':
                img_lines.append(np.hstack((img_line_buffer, appendix_bg_color_space)))
            else:
                img_lines.append(np.hstack((appendix_bg_color_space, img_line_buffer)))
    counter_vein_idx = 0
    for i in vein_joined:
        cv2.imshow(str(counter_vein_idx)+'-th vein', i)
        counter_vein_idx += 1
    img_joined = np.vstack(img_lines)
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.imshow(img_joined, plt.cm.gray)
    ax.set_title('img_joined')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.show()
