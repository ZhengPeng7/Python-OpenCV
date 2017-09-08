import os


def get_images(file_path):
    leaves = []
    path_dir = os.listdir(file_path)
    for i in path_dir:
        child = os.path.join('%s/%s' % (file_path, i))
        leaves.append(child)
    return leaves
