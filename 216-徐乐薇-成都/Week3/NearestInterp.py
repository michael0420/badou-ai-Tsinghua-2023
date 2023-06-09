import cv2
import numpy as np
def function(img):
    height, width, channels = img.shape
    emptyImage = np.zeros((300, 300, channels), np.uint8)
    sh = 300 / height
    sw = 300 / width
    for i in range(300):
        for j in range(300):
            x = int(i / sh + 0.5) #向上取整
            y = int(j / sw + 0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage


img = cv2.imread("lenna.png")
zoom = function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("origin image", img)
cv2.waitKey(0)

