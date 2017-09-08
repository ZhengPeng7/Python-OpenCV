import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import data,filters,segmentation,measure,morphology,color
import cv2
import os  


def splite_img(image):
    """
    Args:
        image - leaves image to be splited
    Ret:
        a list that contains splited images
    """
    
    img=cv2.imread(image)  

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    ret,thresh2=cv2.threshold(GrayImage,165,255,cv2.THRESH_BINARY_INV) 

    thresh =filters.threshold_otsu(thresh2) #阈值分割
    bw =morphology.closing(thresh2 > thresh, morphology.square(3)) #闭运算

    cleared = bw.copy()  #复制
    segmentation.clear_border(cleared)  #清除与边界相连的目标物

    label_image =measure.label(cleared)  #连通区域标记
    borders = np.logical_xor(bw, cleared) #异或
    label_image[borders] = -1
    image_label_overlay =color.label2rgb(label_image, image=thresh2) #不同标记用不同颜色显示

    # fig,(ax0,ax1)= plt.subplots(1,2, figsize=(8, 6))
    # ax0.imshow(cleared,plt.cm.gray)
    # ax1.imshow(image_label_overlay)

    rec=[]
    for region in measure.regionprops(label_image): #循环得到每一个连通区域属性集    
        #忽略小区域
        if region.area < 2000 or region.area>1e6:
            continue
        # print(region.bbox)
        # print(region.centroid)

        #绘制外包矩形
        minr, minc, maxr, maxc = region.bbox
        region_crop = [minr, maxr, minc, maxc]
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        # ax1.add_patch(rect)
        rec.append(region_crop)
    # print(rec)

    rec1 = [x for x in rec if (x[0] <=1000) and ((x[1]-x[0])/(x[3]-x[2])<10) and ((x[1]-x[0])/(x[3]-x[2])>0.1)]
    rec2 = [x for x in rec if (x[0] > 1000) and ((x[1]-x[0])/(x[3]-x[2])<10) and ((x[1]-x[0])/(x[3]-x[2])>0.1)]
    rec1 = sorted(rec1, key=lambda x: x[2])
    rec2 = sorted(rec2, key=lambda x: x[2])
    rec3 = rec1 + rec2
    # print(rec3)

    # fig.tight_layout()
    # plt.show()
    a = []
    for x in rec3:
        crop_img = img[x[0]:x[1], x[2]:x[3]]
        a.append(crop_img)
        # cv2.imshow('cropped', crop_img)
        # cv2.waitKey(0)
    return a



def get_imlist(path):  
    """ 
    Returns a list of filenames for all jpg images in a directory. """  
          
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.JPG')]



def split_save_images(img_path, save_path):
    '''
    Args: img_path : for example - './split_before/'
          save_path : for example - './split_after/'
    
    '''
    images_list = get_imlist(img_path)
    f = os.listdir(img_path)
    print(f)
    print(images_list)

    for d in range(len(images_list)):
        print(images_list[d]+' is being splited ......')
        seg =splite_img(images_list[d])

        for x in range(len(seg)):
            cv2.imwrite(save_path+"seg" +f[d]+'_%x'%x + ".jpg", seg[x])
        print(images_list[d]+ ' is done')
    print('the whole process is done')