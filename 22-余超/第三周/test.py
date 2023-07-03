from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
if __name__ =='__main__':
    im = Image.open('lenna.jpg')
    img = np.asarray(im)
    dst0=Nearest_Neighbor(img,(1500,1800))
    dst1=bilinear_interpolation(img,(1500,1800))
    plt.subplot(221)
    plt.imshow("最邻近插值"，dst0)
    plt.subplot(222)
    plt.imshow("双线性插值",dst1)
    im1 = Image.open('lenna.jpg')
    img1 = np.asarray(im1)
    dst2=histogram_equalization1(img1)
    plt.imshow("直方图均衡化",dst2)
    
