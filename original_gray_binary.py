# -*- coding:utf-8 -*-
'''
@junwei song
image gray binary
'''
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# gray
image = cv2.imread("lenna.png")
h,w = image.shape[:2]  # grab image high and wide
image_gray = np.zeros([h,w],image.dtype) #create the black background and the same dimension
for i in range(h):
    for j in range(w):
        mm = image[i,j]             # get current high and wide position BGR
        image_gray[[i,j]] = int(mm[0]*0.11+mm[1]*0.59 +mm[2]*0.3) # convert BGR into gray
print(image_gray)
print("image show gray: %s" %image_gray)
cv2.imshow("image show gray",image_gray)


# draw the first subimage the original image
plt.subplot(221)
image = plt.imread("lenna.png")
plt.imshow(image)
print("---image lenna---")
print(image)
print(image.shape)
print("the original image shape length:{}".format(len(image.shape)))

# draw the second subimage the image gray
image_gray = rgb2gray(image)
plt.subplot(222)
plt.imshow(image_gray,cmap='gray')
print("---image gray---")
print(image_gray)
print(image_gray.shape)
print("the image_gray shape length: {}".format(len(image_gray.shape)))

# the binary process
# rows,cols = image_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if (image_gray[i,j] < 0.5):
#             image_gray[i,j] = 0
#         else:
#             image_gray[i,j] = 1

image_binary = np.where(image_gray >=0.5,1,0)
print("---image_binary---")
print(image_binary)
print(image_binary.shape)
print("the image_binary shape length: {}".format(len(image_binary.shape)))

#draw the third subimage the bianry image
plt.subplot(223)
plt.imshow(image_binary,cmap='gray')
plt.show()




