import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

if __name__ == '__main__':
    # 1 图像灰度化(方式一) ：调用skimage的rgb2gray 接口
    # 这种方式处理方便，不需要手动计算。
    img = cv2.imread("lenna.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 调用opencv的cvtColor接口,输出的像素范围是0-255

    # 图像灰度化（方式二）：手动将 RGB 三通道各自的值取平均得到灰度值。如果原始图片并没有被正常地归一化到 [0, 1] 的范围内，
    # 那么要根据原始图像的数据类型对其进行最大值的修正。如果是图像的数据类型为 uint8 类型，最大值是 255；如果是 float 类型，最大值是 1。
    # 该方式需要手动计算，但是灵活度高，可以灵活处理不同数据类型的图像。
    # picpath = 'lenna.png'
    # img = plt.imread(picpath)  # 读取的图片存储格式是0到1的浮点数
    # if picpath[-4:] == '.png':   # 如果是png图像，需要将像素值扩展到0-255
    #     img = img * 255   # 扩展范围到0-255
    # img = img.mean(axis = -1)  # 取平均值，得到灰度图像

    #图像灰度化（方式三）：手动写接口
    # image = cv2.imread("lenna.png")
    # h, w = image.shape[:2]  # 获取图片高、宽
    # img = np.zeros([h, w], image.dtype)  # 创建单通道图片，高和宽与原图相同
    # for i in range(h):
    #     for j in range(w):
    #         m = image[i, j]  # 高和宽的BGR坐标
    #         img[i, j] = int(
    #             m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # BGR->灰度化计算公式，m[0]为B，m[1]为G，m[2]为R，cv2是BGR读取顺序，赋给新图像
    # cv2.imshow("image_gray", img)  # 显示灰度化后的图片

    # 2 高斯平滑
    # 2.1. 设置高斯核参数
    sigma = 0.5  # 标准差，可调
    dim = int(np.round(6 * sigma + 1))  # 根据标准差求高斯核是几乘几的，也就是维度
    if dim % 2 == 0:  # 最好是奇数,不是的话加一
        dim += 1
    # 2.2. 生成高斯核,可以调高斯滤波接口
    Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核，这是数组不是列表了
    tmp = [i-dim//2 for i in range(dim)]  # 生成一个序列
    n1 = 1/(2*math.pi*sigma**2)  # 计算高斯核
    n2 = -1/(2*sigma**2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    # 2.3. 边缘填补
    tmp = int(dim/2) #或tmp = dim//2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')  # 边缘填补，img为要填充的数据，第二个参数表示在行列维度上各两方向要填充的像素数量，constant表示边缘填补，边缘填补的值为0
    # 2.4. 高斯平滑
    dx, dy = img.shape #扩展后的img
    img_new = np.zeros(img.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*Gaussian_filter)
    # 2.5. 显示图像
    plt.figure(1) # 创建图像窗口
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')  # 关闭坐标轴

    # 3 sobel检测边缘，求梯度（检测图像中的水平、垂直和对角边缘）
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # x方向
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) # y方向
    img_tidu_x = np.zeros(img_new.shape) # 存储梯度图像
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # x方向
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # y方向
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2) # 梯度幅值
    img_tidu_x[img_tidu_x == 0] = 0.00000001 # 为了后面计算角度不会出错，将矩阵中零值置为一个极小值
    angle = img_tidu_y / img_tidu_x # 梯度方向
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray') # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶（可不加）
    plt.axis('off')

    # 4 非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域（3*3）矩阵。左边冒号 i-1:i+2 表示对矩阵的第一个维度（行）进行切片，从第i−1 行取到第i+2 行（不包括第i+2 行）。
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 5 双阈值检测，连接边缘。遍历所有一定是边的点，进栈,查看8邻域是否存在有可能是边的点，进栈
    # 依次遍历栈内的像素点，判断其周围8个像素点中是否有弱边缘像素点，
    # 若有则添加到栈中，等到所有连通的像素点被检测完后，将这些像素点标记为边缘并从栈中弹出，
    # 同时继续处理新加入栈中的像素点，最终形成完整的轮廓，实现边缘的连通。
    lower_boundary = img_tidu.mean() * 0.5 # 这里我设置低阈值是平均值的0.5倍
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = [] # 存储坐标
    for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了，从1开始，到倒数第二个结束
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
                img_yizhi[i, j] = 255 # 这个像素点标记为边缘
                zhan.append([i, j]) # 进栈
            elif img_yizhi[i, j] <= lower_boundary:  # 舍
                img_yizhi[i, j] = 0

    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 强边缘点出栈的坐标
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2] # 8邻域
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary): # 如果这个点在高低阈值之间，取为边缘点
            img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])    # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_yizhi[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_yizhi[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])

    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0

    # 画图
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off') # 不显示坐标轴
    plt.show()