# 椒盐噪声  指定像素点要么为0要么为255
# 参数为原图、信噪比
def salt_pepper(src_img,percentage):
    NoiseImg=src_img
    NoiseNum=int(percentage*src_img[0]*src_img[1])
    for i in range(NoiseNum):
        #取x Y坐标方向上NoiseNum个随机点
        randX=random.randint(0,NoiseImg.shape[0]-1)
        randY=random.randint(0,NoiseImg.shape[1]-1)
#         random.random()为取0到1之间的随机浮点数
        if random.random<0.5:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg 
