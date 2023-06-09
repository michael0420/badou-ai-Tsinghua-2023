### 噪声与滤波
1. 噪声
    1. 定义：临近像素点之间的突变，在图形学里是中性词
    2. 噪声可以去除，也可以增加，如马赛克特效
    3. 增加噪声要解决2个问题：
        1. 哪些像素点要变？
        2. 变成多少？
    4. 应用：
        1. 马赛克方式
            1. 减少像素点个数
            2. 加噪声，可实现局部马赛克
        2. 库方法调用：util.random_noise(img, mode='gaussian')
2. 典型噪声
    1. 高斯噪声
        1. 区分高斯噪声和白噪声：白噪声指采样样本之间不相关
        2. 产生原因： 自然客观、硬件原因
        3. 如何增加高斯噪声？
            1. 输出像素值公式：Pout = Pin + random.gauss，其中random.gauss 通过输入 sigma(一般设成0.8) 和 mean 确认
            2. 循环每个像素点，random.gauss有时为0，即原像素点，这样整体像素点的噪声就会符合高斯分布
    2. 椒盐噪声/脉冲噪声
        1. 椒盐 = 黑白 = 0/255，黑白色的杂点
        2. 对于彩色图像，表现为单个像素在 BGR 3个通道中随机出现255或0
        3. 产生原因：硬件原因，通信时出错，部分像素值在传输时丢失(丢失至0或255，而不是部分丢失)，比如影像讯号受到突如其来的强烈干扰
        4. 如何增加椒盐噪声？
            1. 指定信噪比SNR，取值在0-1之间
            2. 计算出需要加噪的像素点值：NP = SP * SNR，SP是总像素点数
            3. 随机获得每个NP的位置P(i, j)
            4. 指定像素值为255或0
            5. 重复NP此步骤3和4
    3. 其他噪声
        1. 泊松噪声
        2. 乘性噪声
        3. 瑞利噪声
        4. 伽马噪声
3. 滤波
    1. 目标：
        1. 消除图像中混入的噪声
        2. 抽取图像特征，识别原图像的边缘，与噪声区别开，不损坏图像轮廓和边缘
        3. 需达到更好的视觉效果
        4. 方式：卷积滑动窗口
        5. 滤波是否实时修改参与运算的像素点值，取决于需求场景，一般保留原值
    2. 常见滤波手段
        1. 均值滤波
            1. 图像重置中的最常用手段，没有之一
            2. 遍历处理每个图像，取周围像素点的均值，卷积核是全1矩阵，需要除以矩阵点数
            3. 迭代次数越多，消除尖锐噪声的效果越明显
            4. 优点：简单粗暴
            5. 缺点：图像产生模糊，尤其景物的边缘和细节
        2. 中值滤波
            1. 消除图像噪声最常见手段之一，特别是消除椒盐噪声
            2. 取窗口的中心值
            3. 优点：对椒盐噪声抑制效果很好，画面清晰度基本保持
            4. 缺点：对高斯噪声的抑制效果不好、
        3. 最大最小值滤波
            1. 如果中心值比外圈最大值max大或比最小值min小，则替换为max或min，否则不变
            2. 算法取值比较保守，一般不常用
4. 图像增强
    1. 作用：有目的的增强整体或局部的某些特征、或抑制某些特征，以便让机器有目的的大量学习
    2. 种类：
        1. 点处理技术：只处理单个像素点
        2. 领域处理技术：像素点 + 周围点，即使用卷积核
    3. 点处理
        1. 线性变换：y = ax + b
            1. a影响对比度
                1. 对比度：y1 - y2，所以是a在影响
                2. a > 1：增强对比度
                3. a < 1：减小对比度
            2. b影响亮度
                1. b > 0：图像变亮
                2. b < 0：图像变暗
        2. 分段线性变换
            1. 一次方程组
            2. 对于某个感兴趣的区域的x，增大或减小其a，从而增大或减小这个区域的对比度
        3. 对数变换：y = c * log(1 + x)
            1. 扩展低灰度部分
            2. 压缩高灰度部分
            3. 目的是突出细节，同时可以很好的压缩像素值变化较大的图像的动态范围(结合log函数图像很容易观察出来)
        4. 幂律变换/伽马变换：y = c * x^γ
            1. 修正漂白或过黑的图片，结合函数图像看的更清晰
            2. γ > 1：处理漂泊的图片，进行灰度压缩
            3. γ < 1：处理过黑图片，对比度增强，使得细节看的更清楚
    4. 领域处理：
        1. 滤波
        2. 直方图均衡化
        3. 翻转、平移、旋转、缩放 等等
5. 特征选择与提取
    1. 卷积解决的问题：卷积负责提取图像中的局部特征
    2. 特征的含义：抽象出来的一类属性
    3. 特征的类型：
        1. 相关特征：可以提升学习算法的效果
        2. 无关特征：不会给算法的效果带来任何提升，是在特定的应用场景下讨论的，可能换一个场景就是相关特征了
        3. 冗余特征：可以由其他相关特征推断出来的
    4. 特征选择
        1. 背景：对一个特定的学习算法来说，哪些特征有效是位置的，因此需要从所有特征中选择出对算法有益的相关特征
        2. 目的：
            1. 降维(即减少特征数量)
            2. 降低学习任务的难度
            3. 提升模型的效率
        3. 定义
            1. 从N个特征中选择M个子特征，并且在M(M <= N)个子特征中，准则函数可以达到最优解
            2. 特征选择想要做的是：选择尽可能少的子特征，模型的效果不会显著下降，并且结果的类别分布尽可能的接近真实的类别分布
        4. 怎么做特征选择？主要包括4个过程(可以用选西瓜类比)
            1. 生成过程：生成候选的特征子集——比如：我的幸运数字是7，每隔7个选一个瓜
                1. 完全搜索：分为穷举搜索和非穷举搜索
                2. 启发式搜索——初步做一些判断是否批量选择/拒绝
                3. 随机搜索
            2. 评价函数：评价特征子集的好坏——比如：选择好吃的、或者选择难吃的、或者选择好看的、或者选择屁股有疤的
            3. 停止条件：决定什么时候该停止——挑到什么时候为止
                1. 达到预定义的最大迭代次数——挑烦了
                2. 达到预定义的最大特征数
                3. 增加/删除任何特征不会产生更好的子集——矬子里拔将军，与4的区别是选出的不一定是最优解
                4. 根据评价函数，产生最优特征子集
            4. 验证过程：特征子集是否有效
    5. 特征提取
        1. 降维（组合不同的属性得到新的属性），改变原来的特征空间，比如高鼻梁+大眼睛组合形成大眼妹属性
        2. 特征选择 vs 特征提取
            1. 相同点：作用都是给特征空间降维
            2. 不同点：
                1. 特征选择是物理过程，从原始特征数据集中选出子集，不改变原始的特征空间
                2. 特征提取是化学过程，通过组合不同的属性形成新属性达到降维，改变了原来的特征空间
        3. 2种方法
            1. 传统：基于图像本身的特征进行提取
            2. 深度学习：基于样本自动训练出区分图像的特征分类器——即卷积
        4. 传统特征提取方法
            1. 主成分分析(PCA)——最常见的传统方法
                1. 通过特征值特征向量对已知矩阵降维，将数据从原始的空间中转换到新的特征空间中
                    1. 降维 = 降低列数，列数代表特征
                    2. 行数代表样本数量
                2. 如何球的新的基(a, b, c)？c可由a, b推得，属于冗余基
                    1. step1 对原始数据零均值化（中心化）
                        1. 中心化含义：变量减去它的均值，使均值为0
                        2. 几何意义：中心化是一个平移过程，平移后使得所有数据的中心是(0, 0)
                        3. 目的：中心化数据之后，计算得到的方向才能比较好的概括原来的数据
                    2. step2 求协方差矩阵
                        1. 一组数据在某一坐标轴上的方差越大，说明坐标点越分散
                        2. 方差大能体现出某个特征是否能体现样本差异
                        3. 协方差
                            1. 定义：度量两个随机变量关系的统计量，即如果有一个随机变量为均值，则此时的协方差即为方差
                            2. >0正相关，<0负相关，=0时不相关，要找的就是=0的两个不相关属性
                            3. 协方差只能是2维的，n维数据要用到n阶对称阵，对角线就是各个维度上的方差
                            4. 协方差矩阵计算的是不同维度之间的协方差，而不是不同样本之间的
                            5. 区分样本矩阵 vs 协方差矩阵：样本矩阵的每行是一个样本，每一列为一个维度
                            6. 如果做了中心化，协方差矩阵有个更简单的计算公式，见PPT
                    3. 对协方差矩阵求特征向量和特征值，这些特征向量组成了新的特征空间
                        1. 本质：把一个基底下的东西变换到另一个基底表示的空间中
                        2. 将特征值降序排列，选择最大的k个，对应的k个特征向量作为列向量组成特征矩阵Wnxki(即变换矩阵)
                        3. 原数据集Xnew，降维后的数据集 = XnewWnxki
                3. PCA优化目标：
                    1. 降维后同一纬度的方差最大
                    2. 不同维度之间的相关性为0，即属性不能由其他属性推出，类似一组向量的特征向量
                4. 模型评价的好坏，k值的确定
                    1. 通过特征值的计算，得到主成分所占的百分比，用来衡量模型的好坏
                    2. 前k个特征值保留下的信息量计算公式见PPT
                    3. 前k个特征值在所有特征值里的百分比，数值大小不代表好坏，要根据需求
                5. 三方库
                    1. 方法调用：sklearn：sklearn.decomposition.PCA
                    2. 经典鸢尾花实例: sklearn.datasets.base.load_iris，从图像能看出降低特征维度依然能区分，三种花的坐标能区分开，而不是混在一起
                6. PCA优点
                    1. 无参数，最后的结果只与数据相关，与用户独立
                    2. 可对新主元向量进行重要性排序，选取前面最重要的部分，最大程序保持原有数据信息
                    3. 计算方法简单，易于计算机实现
                7. PCA缺点
                    1. 用户的先验知识无法通过参数化对结果进行干预，效率不高
