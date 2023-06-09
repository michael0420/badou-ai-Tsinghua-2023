### 尺度不变特征变换（SIFT）
1. 问题引入
    1. 两张同一物体的不同角度的照片拼接成一个大照片
        1. 分别提取待拼接图片的特征点
        2. 拟合透视矩阵得到不同场景下的同一物体，进行叠放
    2. 找到物体关键点的算法——SIFT算法
2. SIFT（尺度不变特征变换）
    1. 全称：Scale Invariant Feature Transform
    2. 概念：Sift提取图像的局部特征，在尺度空间寻找极值点，并提取出其位置、尺度、方向信息，共3个方面
    3. 应用范围：物体辨别、机器人地图感知与导航、影像拼接、3D模型建立、手势识别、影像追踪等关键点检测场景
    4. 特点：
        1. 对视角变化、噪声等也存在一定程度的稳定性
            1. Sift所查找的关键点都是一些十分突出，不会因光照，仿射变换和噪声等因素而变换的“稳定”特征点，如角点、边缘点、暗区的亮点以及亮区的暗点等
        2. 独特性，信息量丰富，适用于在海量特征数据中进行快速，准确的匹配
        3. 多量性，即使少数几个物体也可以产生大量的 Sift 特征向量
        4. ！！！商用收费！！！，在opencv > 3.4.3中不再提供
    5. 步骤
        1. 生成高斯差分金字塔（DOG金字塔），尺度空间构建
            1. 尺度空间
                1. 与尺寸区分开，尺寸指的是512 * 512个像素点
                2. 目的：把图片弄成不同的分辨率，模拟人眼近大远小的效果
                3. 通俗地说，尺度空间，就相当于一个图片需要获得多少分辨率的量级。如果把一个图片从原始分辨率不停的对其分辨率进行减少，然后将这些图片摞在一起，可以看成一个四棱锥的样式，这个东西就叫做图像金字塔。
            2. 图像金字塔
                1. 多分辨率尺度解构一张图
                2. 高分辨率图像放在底部，塔尖上是1个像素点的图像
                3. 主要步骤：
                    1. 高斯滤波降噪，平滑图像
                    2. 对平滑图像进行抽象（采样）
                        1. 上采样：分辨率逐级升高
                        2. 下采样：分辨率逐级降低
                    3. 只能模拟近大远小
                4. 进化成高斯金字塔
                    1. 各尺度下图像的模糊度逐渐变大
                    2. 目的：模拟人眼远近模糊程度的变化
                    3. 是构建DOG金字塔构建的基础；DOG金字塔是尺度空间构建的基础
                    4. 概念：高斯金字塔并不是一个金字塔，而是有很多组（Octave）金字塔构成，并且每组金字塔都包含若干层（Interval）
                    5. 构建过程：
                        1. 先将原图像扩大一倍之后作为高斯金字塔的第1组第1层，将第1组第1层图像经高斯卷积之后作为第1组金字塔的第2层。
                        2. 将 σ(一般取1.6，经验数字) 乘以一个比例系数k，得到一个新的平滑因子σ=k*σ，用它来平滑第1组第2层图像，结果图像作为第3层。
                        3. 如此这般重复，最后得到L层图像，在同一组中，每一层图像的尺寸都是一样的，只是平滑系数不一样。它们对应的平滑系数分别为：0，σ，kσ，k^2σ,k^3σ……k^(L-2)σ。
                        4. 构建其他组
                            1. 前3步构建出了一组高斯金字塔。
                            2. 将第1组倒数第三层图像作比例因子为2的降采样（尺寸减半），得到的图像作为第2组的第1层，然后对第2组的第1层图像做平滑因子为σ的高斯平滑，得到第2组的第2层，就像步骤2中一样，如此得到第2组的L层图像，同组内它们的尺寸是一样的，对应的平滑系数分别为：0，σ，kσ，k^2σ,k^3σ……k^(L-2)σ。但是在尺寸方面第2组是第1组图像的一半。
                            3. 经过科学实验，倒数第三层的分辨率比较接近原图。
                        5. 反复执行4，一共得到O组，每组L层，共计O*L个图像，这些图像一起就构成了高斯金字塔
                    6. 特点：
                        1. 在同一组内，不同层图像的尺寸是一样的，后一层图像的高斯平滑因子σ是前一层图像平滑因子的k倍
                        2. 在不同组间，后一组第一个图像是前一组倒数第三个图像的二分之一采样，图像大小是前一组的一半
                    6. 构建尺度空间：坐标系(O，L)
                        1. 在高斯金字塔中一共生成O组L层不同尺度的图像，这两个量合起来（O，L）就构成了高斯金字塔的尺度空间
                        2. 以高斯金字塔的组O作为二维坐标系的一个坐标，不同层L作为另一个坐标，则给定的一组坐标（O,L）就可以唯一确定高斯金字塔中的一幅图像
                        3. 在计算机中以二维数组的形式存储
                    7. 算法原理：
                        1. SIFT算法在构建尺度空间时候采取高斯核函数进行滤波，使原始图像保存最多的细节特征，经过高斯滤波后细节特征逐渐减少来模拟大尺度情况下的特征表示。
                        2. 其实尺度空间图像生成就是当前图像与不同尺度核参数σ进行卷积运算后产生的图像
                5. 进化成高斯差分金字塔，即DOG（Difference of Gaussian）金字塔
                    1. 构建过程（相对简单）：
                        1. DOG金字塔的第1组第1层是由高斯金字塔的第1组第2层减第1组第1层得到的。以此类推，逐组逐层生成每一个差分图像，所有差分图像构成差分金字塔
                        2. 概括为DOG金字塔的第o组第l层图像是由高斯金字塔的第o组第l+1层减第o组第l层得到的
                        3. 说人话就是两两相减
                    2. 构建目的：
                        1. 找特征点/关键点：相同的点被减没了，即去掉不同情况下经过高斯模糊没有变化的点，这些点就是非特征值的点（卷积/滤波的本质是提取局部特征，高斯滤波本质提取的是边缘）
                    3. 得到的结果：
                        1. 人眼见到的是全黑图，但打印出矩阵的矩阵非全零矩阵
                        2. 归一化处理后可见轮廓特征，即SIFT找到的稳定特征（这一步不是算法需要的，只是为了给人看）
        2. 空间极值点检测（关键点的初步查探）——进一步的筛选，进行非极大值抑制NMS的过程
            1. 尺度空间极值检测——找极值
            2. 8邻域以及上下对应的9个点，共26个点比较，以确保在尺度空间和二维图像空间都检测到极值点。类似非极大值抑制NMS
            3. 一组中的第一层图和最后一层图的计算方法：再用高斯模糊，在高斯金字塔中多模糊出三张来凑数，所以在DOG中多出两张
                1. DOG金字塔上下各多一张，共多两张
                2. 高斯金字塔需要比DOG金字塔多一张（因为是做减法），所以高斯金字塔要多3张
            4. 高斯金字塔的k值设定：k = 2 ^ (1/s)
                1. s：每组图像中检测s个尺度的极值点。（实际计算时，s通常在3到5之间）
                2. s的意思是将来我们在差分高斯金字塔中求极值点的时候，我们要求s层点。
                3. 高斯金字塔共包含O组图像，每组图像有 s + 3 层，O = log2(min{M, N}) - 3 // M,N为原始图像的行数和列数
        3. 稳定关键点的精确定位（可选步骤）
            1. why：DOG值对噪声和边缘比较敏感，所以在第2步的尺度空间中检测到的局部极值点还要经过进一步的筛选，去除不稳定和错误检测出的极值点。
            2. how：利用阈值的方法来限制，在opencv中为contrastThreshold，人为设置此参数，如果值选取的不好会带来副作用
            3. 这一步并不是必要的，为了避免副作用可以跳过这一步
            4. 至此，SIFT特征的位置、尺度都找到了。
        4. 稳定关键点方向信息分配
            1. 方法：
                1. 获取关键点所在尺度空间的邻域，然后计算该区域的梯度和方向，根据计算得到的结果创建方向直方图。
                2. 直方图的峰值为主方向的参数，其他高于主方向80%的方向被判定为辅助方向，这里的80%也是可以根据需求改变的。
            2. 任一关键点的梯度公式：
                1. 梯度幅值（看起来很复杂的数学公式），y方向的差值与x方向差值的几何距离，勾股定理
                2. 梯度方向（看起来很复杂的数学公式），y方向的差值 / x方向的差值
        5. 关键点描述
            1. 要描述出来的东西(what)：位置、尺度、方向、周围对其有贡献的邻域点
            2. 如何描述(how)：4 * 4 * 8 共128维的向量描述中心关键点效果最有(由实验得出)（不变性与独特性）
            3. 至此完成了关键点描述，可以进行两张图的关键点匹配了
        6. 特征点匹配
            1. 特征点的匹配是通过计算两组特征点的128维的关键点的欧式距离实现的。
            2. 欧式距离越小，则相似度越高，当欧式距离小于设定的阈值时，可以判定为匹配成功。
            3. 步骤：
                1. 分别对模板图（参考图，reference image）和实时图（观测图，observation image）建立关键点描述子集合。目标的识别是通过两点集内关键点描述子的比对来完成。具有128维的关键点描述子的相似性度量采用欧式距离。
                2. 匹配可采取穷举法完成。

### OpenCV 算法解析
1. OpenCV
    1. 简介
        1. 开源计算机视觉库，http://opencv.org
        2. 用 C 和 C++ 编写，跨操作系统
        3. 高效、多线程
        4. 坑：读入的彩色图片BGR顺序，但存储是RGB
            ```
            #opencv读入的矩阵是BGR，如果想转为RGB，可以这么转
            img4 = cv2.imread('1.jpg')
            img4 = cv2.cvtColor(img4,cv2.COLOR_BGR2RGB)
            ```
    2. 横向比较
        1. 读取通道BGR
        2. 除了PIL读入的图片是img类，其他库读进来的图片都是numpy矩阵
        3. 性能好，速度和全面性都是碾压性的存在
    3. OpenCV常见算法
        1. 图像的基本操作
            1. 读取：cv2.imread()
            2. 显示：cv2.imshow()
            3. 存储：cv2.write()
        2. 彩色像素转化为灰度像素
        3. 图像的几何变换：
            1. 平移
            2. 缩放
            3. 旋转
            4. 插值（最邻近、双线性）
            5. 边缘检测
                1. Sobel
                2. Canny
            6. 图像的二维滤波：cvFilfer2D
        4. 最小二乘法
            1. 线性回归：线性回归表示这些离散的点总体上“最逼近”哪条直线。
            2. 最小二乘法（Least Square Method）
                1. 它通过最小化误差的平方和，寻找数据的最佳函数匹配。
                2. 利用最小二乘法可以简便地求得未知的数据，并使得这些求得的数据与实际数据之间误差的平方和为最小。为什么选平方和？
                    1. 平方可以避免误差正负号抵消
                    2. 平方可以放大大误差，减小小误差
                    3. 高斯对此有一个完美的数学证明，但是这里地方太小写不开了。
                3. 3种误差范数，这里需要选第三种 “2-范数”：
                    1. ∞-范数：残差绝对值的最大值
                        1. 计算量过大
                    2. 1-范数：绝对残差和
                        1. 计算机中绝对值比平方的计算量大，需要先判断符号再运算
                    3. 2-范数：残差平方和 √
                4. 拟合程度
                    1. 通俗来讲，就是我们的拟合函数h(x)与待求解的函数y之间的相似性
                    2. 那么2-范数越小，相似性就比较高
                5. 在最小二乘法的数学式子中，分别对 k 和 b 求偏导并另偏导数为0，求得k和b取什么值能得到极值点。

### RANSAC 思想
1. 简介
    1. 定义：随机采样一致性（random sample consensus）。
    2. 思想：RANSAC是一种思想，一个求解已知模型的参数的框架。它不限定某一特定的问题，可以是计算机视觉的问题，同样也可以是统计数学，甚至可以是经济学领域的模型参数估计问题。
    3. 结果评估：
        1. 它是一种迭代的方法，用来在一组包含离群的被观测数据中估算出数学模型的参数。
        2. RANSAC是一个非确定性算法，在某种意义上说，它会产生一个在一定概率下合理的结果，其允许使用更多次的迭代来使其概率增加。
    4. 过程：
        1. RANSAC的基本假设是 “内群”数据可以通过几组模型参数来叙述其数据分布，而“离群”数据则是不适合模型化的数据。
        2. 数据会受噪声影响,噪声指的是离群，例如从极端的噪声或错误解释有关数据的测量或不正确的假设
        3. RANSAC假定，给定一组（通常很小的）内群，存在一个程序，这个程序可以估算最佳解释或最适用于这一数据模型的参数。
2. RANSAC VS 最小二乘法
    1. 前景：生产实践中的数据往往会有一定的偏差。
    2. 引入例子与目的：
        1. 例如我们知道两个变量X与Y之间呈线性关系，Y=aX+b，我们想确定参数a与b的具体值。
        2. 通过实验，可以得到一组X与Y的测试值。虽然理论上两个未知数的方程只需要两组值即可确认，但由于系统误差的原因，任意取两点算出的a与b的值都不尽相同。
        3. 我们希望的是，最后计算得出的理论模型与测试值的误差最小。
    3. 引入最小二乘法：
        1. 通过计算最小均方差关于参数a、b的偏导数为零时的值。
        2. 事实上，很多情况下，最小二乘法都是线性回归的代名词。
        3. 遗憾的是，最小二乘法只适合于误差较小的情况。
    4. RANSAC 在以下2点优于最小二乘法：
        1. 数据集存在较多误差：
            1. 在模型确定以及最大迭代次数允许的情况下，RANSAC总是能找到最优解。
            2. 对于包含80%误差的数据集，RANSAC的效果远优于直接的最小二乘法。
        2. 算力挑战：由于一张图片中像素点数量大，采用最小二乘法运算量大，计算速度慢。
3. RANSAC 步骤
    1. 输入：
        1. 一组观测数据（往往含有较大的噪声或无效点）
        2. 一个用于解释观测数据的参数化模型，比如 y=ax+b
        3. 一些可信的参数。
    2. 步骤：
        1. 在数据中随机选择几个点设定为内群
        2. 计算适合内群的模型 e.g. y=ax+b ->y=2x+3 y=4x+5
        3. 把其它刚才没选到的点带入刚才建立的模型中，计算是否为内群 e.g. hi=2xi+3->ri（即误差阈值）
        4. 记下内群数量（内群中点的数量）
        5. 重复以上步骤（1-4）
        6. 比较哪次计算中内群数量最多,内群最多的那次所建的模型就是我们所要求的解
    3. 缺点：
        1. 实质：相当于穷举每个模型，问题是能穷举多少次（k）
        3. 要求数学模型已知：不同问题对应的数学模型不同，因此在计算模型参数时方法必定不同
    4. 实操问题：
        1. 一开始的时候我们要随机选择多少点(n)
        2. 要重复做多少次(k)
    5. RANSAC 的参数确定
        1. 假设每个点是真正内群的概率为 w：w = 内群的数目/（内群数目+外群数目）
        2. 通常我们不知道 w 是多少
            1. w^n是所选择的n个点都是内群的机率
            2. 1-w^n 是所选择的n个点至少有一个不是内群的机率
            3. (1 − w^n)^k 是表示重复 k 次，每次都至少有一个点不是内群
            4. 假设算法跑 k 次以后成功的机率是p，得：
                1. 1 − p = (1 − w^n)^k => p = 1 − (1 − w^n)^k
                2. K = log(1-P)/log(1-w^n)
                3. 希望成功率P越高：
                    1. 当n不变时，k越大，则p越大； 当w不变时，n越大，所需的k就越大。而w是不且未知的；
                    2. 通常w未知，所以n 选小一点比较好，这样k（迭代次数）就会比较小。
            5. 迭代停止条件：
                    1. 工程上通常会给k设置一个阈值，例如迭代100次就停止
                    2. 或者迭代多次后内群数量不增加，即停止
    6. RANSAC的优缺点
        1. 优点
            1. 它能鲁棒的估计模型参数，即包容较多噪声。例如，它能从包含大量局外点的数据集中估计出高精度的参数。
        2. 缺点：
            1. 它计算参数的迭代次数没有上限；如果设置迭代次数的上限，得到的结果可能不是最优的结果，甚至可能得到错误的结果。
            2. RANSAC只有一定的概率得到可信的模型，概率与迭代次数成正比。
            3. 它要求设置跟问题相关的阀值，不同问题领域得阈值都不同
            4. RANSAC只能从特定的数据集中估计出一个模型，如果存在两个（或多个）模型，RANSAC不能找到别的模型。
            5. 要求数学模型已知

### 图像相似度比较哈希算法
1. 相似图像搜索的哈希算法有三种:
    1. 均值哈希算法 aHash
    2. 差值哈希算法 dHash
    3. 感知哈希算法 pHash
2. 特点：
    1. 不可逆性
    2. 雪崩性
3. 应用：
    1. 量化图像相似度，利用汉明距离
        1. 汉明距离：两个整数之间的汉明距离指的是这两个数字对应二进制位不同的位置的数目。
4. 均值哈希算法
    1. 步骤：
        1. 缩放
            1. 图片缩放为8*8，保留结构，除去细节（只见森林，不见树木）
                1. 8*8的数据来自于实验科学
            2. 原因：原图数据量太多，会导致计算量太大，如200 * 200的图片，像素点有200 * 200 * 3 = 120000个
        2. 灰度化：转换为灰度图。
        3. 求平均值：计算灰度图所有像素的平均值。
        4. 比较：像素值大于平均值记作1，相反记作0，总共64位（8*8）。
        5. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
        6. 对比指纹：
            1. 将两幅图的指纹对比，计算汉明距离，即两个64位的hash值有多少位是不一样的；需要自己设置汉明距离的临界值
            2. 不相同位数越少，图片越相似。
    2. 算法基于基本假设：图像缩小之后，均值不会相差特别多
    3. 缩放和灰度化会带来误差
5. 差值哈希算法
    1. 差值哈希算法相较于均值哈希算法，前期和后期基本相同，只有中间比较hash有变化。
    2. 步骤：
        1. 缩放：图片缩放为8*9，保留结构，除去细节
            1. 多一列是因为第三步中，本行的后一列与前一列要做插值，9列才能保证减完之后剩下8列
        2. 灰度化：转换为灰度图。但无需计算灰度图像素平均值，与aHash区分
        3. 比较：像素值大于后一个像素值记作1，相反记作0。本行不与下一行对比，每行9个像素，八个差值，有8行，总共64位
        4. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
        5. 对比指纹：将两幅图的指纹对比，计算汉明距离，即两个64位的hash值有多少位是不一样的，不相同位数越少，图片越相似。
6. 感知哈希算法
    1. 较复杂，但不能换来与多付出的代价相匹配的精确度，不常用
6. 三种图像相似度算法的比较：
    1. aHash：均值哈希。速度比较快，但是有时不太精确。
    2. dHash：差值哈希。精确度较高，且速度也非常快。√
    3. pHash：感知哈希。精确度较高，但是速度方面较差一些。
