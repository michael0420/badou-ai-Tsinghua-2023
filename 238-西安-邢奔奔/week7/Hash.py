import cv2
import numpy as np

#差值哈希
def sHash(img):
    img = cv2.resize(img,(8,9),interpolation=cv2.INTER_CUBIC)
    '''
    interpolation=cv2.INTER_CUBIC：插值方法的选择
    这里使用的是 cv2.INTER_CUBIC，表示使用立方插值方法进行调整
    还可以选择其他的插值方法，如 cv2.INTER_LINEAR（双线性插值）或 cv2.INTER_NEAREST（最近邻插值）
    '''
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i,j] > gray[i,j+1]:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str
#均值哈希
def aHash(img):
    img = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    s = 0
    hast_str = ''
    for i in range(8):
        for j in range(8):
            s = s + gray[i,j]
    avg = s / 64

    for i in range(8):
        for j in range(8):
            if gray[i,j] > avg:
                hash_str = hast_str + '1'
            else:
                hast_str = hash_str + '0'
    return hash_str
#不相同的位数越少，图片越相似
def cmpHash(hash1,hash2):
    if len(hash1) != len(hash2):
        return -1
    s = 0
    for i in range(8):
        for j in range(8):
            if hash1[i,j] != hash2[i,j]:
               s += 1
    return s
