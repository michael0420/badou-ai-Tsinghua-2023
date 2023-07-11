import cv2
import random

def GaussNoise(img, means, sigma, percentage):
    gaussImg = img
    noiseNum = int(percentage * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[1] - 1)
        randY = random.randint(0, img.shape[0] - 1)
        gaussImg[randX, randY] = gaussImg[randX, randY] + random.gauss(means, sigma)
        if gaussImg[randX, randY] < 0:
            gaussImg[randX, randY] = 0
        elif gaussImg[randX, randY] > 255:
            gaussImg[randX, randY] = 255
    return gaussImg

img1 = cv2.imread("lenna.png", 0)
img2 = GaussNoise(img1, 2, 4, 0.8)
img = cv2.imread('lenna.png')
img3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("source", img1)
cv2.imshow("gauss", img2)
cv2.imshow("gray", img3)
cv2.waitKey(0)
