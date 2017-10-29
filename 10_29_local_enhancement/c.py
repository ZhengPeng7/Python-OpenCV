import cv2
fn = "0.jpg"
if __name__ == '__main__':
    print('loading %s ...' % fn)
    print('processing...')
    img = cv2.imread(fn)
    w = img.shape[1]
    h = img.shape[0]
    ii = 0
    # let image get darker
    for xi in range(0,w):
        for xj in range(0,h):
            # set the pixel value decrease to 20%
            img[xj,xi,0] = int(img[xj,xi,0]*0.2)
            img[xj,xi,1] = int(img[xj,xi,0]*0.2)
            img[xj,xi,2] = int(img[xj,xi,0]*0.2)
        # show the process
        if xi%10==0:
            print('.')
    cv2.namedWindow('img')
    cv2.imshow('img',img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print('.')
    print('processing')
    for xi in range(0,w):
        for xj in range(0,h):
            # set the pixel value increase to 1020%
            img[xj,xi,0] = int(img[xj,xi,0]*10.2)
            img[xj,xi,1] = int(img[xj,xi,1]*10.2)
            img[xj,xi,2] = int(img[xj,xi,2]*10.2)
            # show the process
        if xi%10==0:
            print('.')
    cv2.namedWindow('img')
    cv2.imshow('img',img)
    cv2.waitKey()
    cv2.destroyAllWindows()
