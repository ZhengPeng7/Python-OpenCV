import cv2
import numpy as np
import get_images
import click_for_points
import save_faces_corrected


# get faces_original
faces_lst = get_images.get_images(r'./jaffe')
print('faces_lst:', faces_lst)
print('len(faces_lst):', len(faces_lst))

# get the axis_criteria
pts_criteria = click_for_points.click_for_points(cv2.imread(faces_lst[0]))
faces_corrected = [cv2.imread(faces_lst[0])]

# correct the following faces_original
counter = 0
for i in faces_lst[1:]:
    img = cv2.imread(i)
    rows, cols, ch = img.shape
    pts_to_be_corrected = click_for_points.click_for_points(img)

    M = cv2.getAffineTransform(pts_to_be_corrected, pts_criteria)

    dst = cv2.warpAffine(img, M, (cols, rows))

    faces_corrected.append(np.array(dst))
    counter += 1
    if not counter % 10:
        print(counter)

# save the faces_corrected into r'./corrected_after'
save_faces_corrected.save_faces_corrected(faces_corrected, faces_lst, path=r'./corrected_after')
