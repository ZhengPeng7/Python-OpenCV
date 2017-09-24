import cv2


def save_faces_corrected(faces_corrected, leaves_source, path=r'./'):
    if path[-1] != '/':
        path += '/'
    for i in range(len(faces_corrected)):
        cv2.imwrite(path + 'corrected_' + leaves_source[i].rpartition('/')[-1], faces_corrected[i])
