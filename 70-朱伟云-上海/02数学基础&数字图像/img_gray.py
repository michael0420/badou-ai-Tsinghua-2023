from PIL import Image

# 打开图片
img = Image.open("pic/1.png");

# 将图片转换为黑白
img = img.convert('L')

# 设定阈值
threshold = 100

# 遍历每个像素点，如果颜色值大于阈值则设为255（白色），否则为0（黑色）
for i in range(img.width):
    for j in range(img.height):
        if img.getpixel((i, j)) > threshold:
            img.putpixel((i, j), 0)
        else:
            img.putpixel((i, j), 255)

            
#保存结果
img.save("pic/1-l.png")


