最近邻插值
优点：
计算量很小，算法简单，速度快。
缺点：
仅使用离待测采样点最近的像素的灰度值作为该采样点的灰度值，而没有考虑其他相邻像素点的影响，因此采样后的灰度值有明显的不连续性，图像质量损失较大，会产生明显的马赛克和锯齿现象。

双线性插值
优点
双线性插值效果比最近邻插值效果好，缩放后的图像质量高。
因为考虑了待采样点周围四个直接邻点对该采样点的相关性影响，基本克服了最近邻插值不连续的缺点。
缺点
计算量稍大，算法更复杂一些，运算时间长。
仅考虑待测样点周围四个点灰度值的影响，而为考虑各邻点间灰度值变化率的影响，因此具有低通滤波器的性质，导致缩放后图像的高频分量受到损失，图像边缘产生一定程度的模糊。
用此方法缩放后的图像与输入图像相比，仍然存在由于插值函数设计考虑不周而产生的图像质量受损的问题。

直方图
直方图均衡的目的是使图像灰度间距拉开或使灰度分布均匀，从而增大对比度，使图像细节清晰，实现图像空间增强
