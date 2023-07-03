# 彩色图的直方图均衡化 equalize-直方图均衡化
def histogram_equalization1(img):
    height,width,channels=img.shape
    dst=np.zeros((height,width,channels),np.uint8)
    (r,g,b)=cv2.split(img)
    r1=cv2.equalizeHist(r)
    g1=cv2.equalizeHist(g)
    b1=cv2.equalizeHist(b)
    dst=cv2.merge((r1,g1,b1))
    return dst 
