# 双线性插值算法 参数同上
def bilinear_interpolation(img,out_dim):
    height,width,channels=img.shape
    dst_h,dst_w=out_dim[0],out_dim[1]
    if height==dst_h and width==out_dim[1]:
         return img
    targetImg=np.zeros((dst_h,dst_w,channels),np.uint8)
# sh、sw 为插值算法的缩放比例 :原图各维度像素点个数/目标图各维度像素点个数
    sh=float (height/dst_h)
    sw=float (width/dst_w)
    for i in range(channels):
        for dst_x in range(dst_h):
            for dst_y in range(dst_w):   
# 几何中心重合。x、y为遍历该点对应缩放前在原图中的坐标值
                x=(dst_x+0.5)*sh-0.5
                y=(dst_y+0.5)*sw-0.5
#         x0、x1、y0、y1为点（x，y）周围临近四个点的横纵坐标
                x0=int(np.floor(x))
                x1=min(x0+1,height-1)
                y0=int(np.floor(y))
                y1=min(y0+1,width-1)
                targetImg[dst_x,dst_y,i]=int((y1-y)*((x1-x)*img[x0,y0,i]+(x-x0)*img[x1,y0,i])+(y-y0)*((x1-x)*img[x0,y1,i]+(x-x0)*img[x1,y1,i]))
    return targetImg
