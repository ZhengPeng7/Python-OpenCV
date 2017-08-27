import skimage.io as io
from skimage import data_dir
import extract_vein_by_hough


img_dir = r'./images/*.jpg'
coll = io.ImageCollection(img_dir)
img = coll.files


for i in img:
    extract_vein_by_hough.extract_vein_by_hough(i)
