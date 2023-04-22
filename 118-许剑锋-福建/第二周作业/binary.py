'''
二值化
'''

import cv2
import numpy as np

def showImage(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()


def binary(img):
    target = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    width = img.shape[0]
    height = img.shape[1]

    count = 0
    for i in range(width):
        for j in range(height):
            count += img[i][j]
    avg = count // (width * height)
    print('avg', avg)
    for i in range(width):
        for j in range(height):
            if img[i][j] >= avg:
                target[i][j] = 255
            else:
                target[i][j] = 0
    return target


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binaryImage = binary(gray)
    showImage(binaryImage, 'binaryImage')