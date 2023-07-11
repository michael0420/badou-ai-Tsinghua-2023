import cv2
import random


def Noise(img, percentage):
    noiseImg = img
    noiseNum = int(percentage * noiseImg.shape[0] * noiseImg.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, noiseImg.shape[0] - 1)
        randY = random.randint(0, noiseImg.shape[1] - 1)
        if random.random() <= 0.5:
            noiseImg[randX, randX] = 0
        else:
            noiseImg[randX, randY] = 255
    return noiseImg

img1 = cv2.imread("lenna.png", 0)
img1 = Noise(img1, 0.2)
img2 = cv2.imread("lenna.png")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cv2.imshow("source", img2)
cv2.imshow("noise", img1)
cv2.waitKey(0)
