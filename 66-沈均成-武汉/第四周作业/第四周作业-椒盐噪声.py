import cv2
import random


def PepperSaltNoise(src,percetage):
    NoiseImg=src
    # shape[0]表示矩阵的行数，shape[1]表示矩阵的列数
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
	    #矩阵下标是从0开始，比如256矩阵，就是0-255
	    randX=random.randint(0,src.shape[0]-1)
	    randY=random.randint(0,src.shape[1]-1)
	    #random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
	    if random.random()<=0.5:
	    	NoiseImg[randX,randY]=0
	    else:
	    	NoiseImg[randX,randY]=255
    return NoiseImg

# cv2.IMREAD_COLOR：加载彩色图片，这个是默认参数，可以直接写1。
# cv2.IMREAD_GRAYSCALE：以灰度模式加载图片，可以直接写0。
# cv2.IMREAD_UNCHANGED：包括alpha，可以直接写-1
img=cv2.imread('lenna.png',0)
img1=PepperSaltNoise(img,0.2)
#经过函数GaussianNoise后，img被修改了，需要重新读取一下，才是原图
img=cv2.imread('lenna.png',0)

cv2.imshow('source',img)
cv2.imshow('lenna_PepperandSalt',img1)
cv2.waitKey(0)