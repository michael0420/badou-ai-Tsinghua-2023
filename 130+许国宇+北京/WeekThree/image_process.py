import cv2
import numpy as np
import matplotlib.pyplot as plt



class ImageProcess:
    def __init__(self,image_path):
        self.image=cv2.imread(image_path)
        self.image_plt=plt.imread(imagePath)
        self.image_width=self.image.shape[:2][0]
        self.image_height=self.image.shape[:2][1]
        self.image_channels=self.image.shape[2]
        self.image_gray_ok=False
        self.image_thresholding_ok=False
        self.image_nearestInterp_ok=False
        self.image_bilinearInterp_ok=False
        self.image_histogram_ok=False
        self.image_histEqualized_ok=False
    #最临近插值
    def nearestInterp(self,dst_width,dst_height):
        self.image_nearestInterp=np.zeros([dst_width,dst_height,self.image_channels],np.uint8)
        self.image_nearestInterp_plt=np.zeros([dst_width,dst_height,self.image_channels],self.image_plt.dtype)
        #计算比例关系
        scale_width=dst_width/self.image_width
        scale_height=dst_height/self.image_height
        for i in range(dst_width):
            for j in range(dst_height):
                x=int(i/scale_width+0.5)
                y=int(j/scale_height+0.5)
                self.image_nearestInterp[i,j]=self.image[x,y]
                self.image_nearestInterp_plt[i,j]=self.image_plt[x,y]
        self.image_nearestInterp_ok=True
    #bilinear Interpolation
    def bilinearInterp(self,dst_width,dst_height):
        self.image_bilinearInterp=np.zeros([dst_width,dst_height,self.image_channels],np.uint8)
        self.image_bilinearInterp_plt=np.zeros([dst_width,dst_height,self.image_channels],self.image_plt.dtype)
        scale_width=self.image_width/dst_width
        scale_height=self.image_height/dst_height
        for channel in range(self.image_channels):
            for i in range(dst_width):
                for j in range(dst_height):
                    #align images
                    src_x=(i+0.5)*scale_width-0.5
                    src_y=(j+0.5)*scale_height-0.5
                    #get the two adjacent points
                    src_x0=int(np.floor(src_x))
                    src_x1=min(src_x0+1,self.image_width-1)

                    src_y0=int(np.floor(src_y))
                    src_y1=min(src_y0+1,self.image_height-1)

                    r1=(src_x1-src_x)*self.image[src_x0,src_y0,channel]+(src_x-src_x0)*self.image[src_x1,src_y0,channel]

                    r2=(src_x1-src_x)*self.image[src_x0,src_y1,channel]+(src_x-src_x0)*self.image[src_x1,src_y1,channel]

                    self.image_bilinearInterp[i,j,channel]=int((src_y1-src_y)*r1+(src_y-src_y0)*r2)
                    #plt the sequence of b-g-r of the plt program is r-g-b,not the same as the sequence of b-g-r of cv2 program is b-g-r
                    if channel==0:
                        r1_plt = (src_x1 - src_x) * self.image_plt[src_x0, src_y0, channel+2] + (src_x - src_x0) * \
                                 self.image_plt[src_x1, src_y0, channel+2]
                        r2_plt = (src_x1 - src_x) * self.image_plt[src_x0, src_y1, channel+2] + (src_x - src_x0) * \
                                 self.image_plt[src_x1, src_y1, channel+2]
                        self.image_bilinearInterp_plt[i,j,channel+2]=round((src_y1-src_y)*r1_plt+(src_y-src_y0)*r2_plt,2)
                    elif channel==2:
                        r1_plt = (src_x1 - src_x) * self.image_plt[src_x0, src_y0, channel-2] + (src_x - src_x0) * \
                                 self.image_plt[src_x1, src_y0, channel-2]
                        r2_plt = (src_x1 - src_x) * self.image_plt[src_x0, src_y1, channel-2] + (src_x - src_x0) * \
                                 self.image_plt[src_x1, src_y1, channel-2]
                        self.image_bilinearInterp_plt[i, j, channel-2] = round((src_y1 - src_y) * r1_plt + (src_y - src_y0) * r2_plt, 2)
                    else:
                        r1_plt = (src_x1 - src_x) * self.image_plt[src_x0, src_y0, channel] + (src_x - src_x0) * \
                                 self.image_plt[src_x1, src_y0, channel]
                        r2_plt = (src_x1 - src_x) * self.image_plt[src_x0, src_y1, channel] + (src_x - src_x0) * \
                                 self.image_plt[src_x1, src_y1, channel]
                        self.image_bilinearInterp_plt[i, j, channel] = round(
                            (src_y1 - src_y) * r1_plt + (src_y - src_y0) * r2_plt, 2)

        self.image_bilinearInterp_ok=True
    #histogram
    def handleHistogram(self):
        self.image_histogram_ok = False
        #first get the grayscale image
        self.handleGray()
        self.image_gray_ok=False
        #second open up gray statistical space
        self.gray_count=np.zeros(256)
        #third iterate through the image
        for i in range(self.image_width):
            for j in range(self.image_height):
                self.gray_count[self.image_gray[i,j]]+=1
        self.image_histogram_ok=True
    def histogram(self):
        self.image_histogram_ok = False
        self.gray_count = np.zeros(256)
        # first get the grayscale image
        self.handleGray()
        self.image_gray_ok = False
        #second get the histogram
        hist=cv2.calcHist([self.image_gray],[0],None,[256],[0,256])
        self.gray_count=hist
        self.image_histogram_ok = True
    def handleHistEqualization(self):
        self.image_histEqualization=np.zeros([self.image_width,self.image_height],np.uint8)
        self.image_histEqualization_plt=np.zeros([self.image_width,self.image_height],dtype=self.image_plt.dtype)
        self.handleHistogram()
        self.image_histogram_ok=False
        #calculate the sum of pixes
        sumPixesOfImage=self.image_width*self.image_height
        #identify the points in the gray image where the gray value is not zero
        grayPercentage=np.zeros(256)
        sumOfGrayPercentage=np.zeros(256)
        equalizedHist=np.zeros(256)
        for i in range(256):
            #calculate the percentage of each gray value
            grayPercentage[i]=self.gray_count[i]/sumPixesOfImage
            for j in range(i+1):
                #calculate the sum of percentage
                sumOfGrayPercentage[i]+=grayPercentage[j]
            equalizedHist[i]=max(int(sumOfGrayPercentage[i]*256-1),0)
        for m in range(self.image_width):
            for n in range(self.image_height):
                self.image_histEqualization[m,n]=equalizedHist[self.image_gray[m,n]]
                self.image_histEqualization_plt[m,n]=equalizedHist[self.image_gray[m,n]]
                self.image_histEqualization_plt[m, n]=self.image_histEqualization_plt[m, n]/256
        self.image_histEqualized_ok=True
    def histEqualization(self):
        self.image_histEqualization = np.zeros([self.image_width, self.image_height], np.uint8)
        self.handleGray()
        self.image_gray_ok = False
        self.image_histEqualization=cv2.equalizeHist(self.image_gray)
        self.image_histEqualized_ok = True
    #手动灰度处理
    def handleGray(self):
        self.image_gray = np.zeros([self.image_width, self.image_height], self.image.dtype)
        self.image_gray_plt = np.zeros([self.image_width, self.image_height], self.image_plt.dtype)
        for i in range(self.image_width):
            for j in range(self.image_height):
                m=self.image[i,j]
                n=self.image_plt[i,j]
                self.image_gray[i,j]=int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
                self.image_gray_plt[i,j]=(n[0]*0.3+n[1]*0.59+n[2]*0.11)
        self.image_gray_ok=True
    #灰度处理
    def gray(self):
        self.image_gray = np.zeros([self.image_width, self.image_height], self.image.dtype)
        self.image_gray_plt = np.zeros([self.image_width, self.image_height], self.image_plt.dtype)
        self.image_gray=cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        self.image_gray_plt=cv2.cvtColor(self.image_plt,cv2.COLOR_BGR2GRAY)
        self.image_gray_ok = True
    #manual binarization
    def handleThresholding(self):
        self.handleGray()
        self.image_gray_ok=False
        self.image_thresholding=np.zeros([self.image_width,self.image_height])
        self.image_thresholding_plt=np.zeros([self.image_width,self.image_height],dtype=self.image_plt.dtype)
        for i in range(self.image_width):
            for j in range(self.image_height):
                if self.image_gray[i, j]/256<=0.5:
                    self.image_thresholding[i,j]=0
                else:
                    self.image_thresholding[i,j]=1

                if self.image_gray_plt[i, j]<=0.5:
                    self.image_thresholding_plt[i,j]=0
                else:
                    self.image_thresholding_plt[i,j]=1
        self.image_thresholding_ok=True
    #二值化处理
    def thresholding(self):
        self.image_thresholding = np.zeros([self.image_width, self.image_height])
        self.image_thresholding_plt = np.zeros([self.image_width, self.image_height])
        self.image_thresholding_ok=True
        return self.image_thresholding
    def thresholdingPlt(self):
        self.image_thresholding = np.zeros([self.image_width, self.image_height])
        self.image_thresholding_plt = np.zeros([self.image_width, self.image_height])
        self.image_thresholding_ok=True
        return self.image_thresholding_plt
    #使用cv2函数显示结果
    def cv2show(self):
        if (self.image_gray_ok==True):
            cv2.imshow("The Origin Picture",self.image)
            cv2.imshow("The Gray picture",self.image_gray)
            cv2.waitKey()
        elif(self.image_thresholding_ok==True):
            cv2.imshow("The Origin Picture", self.image)
            cv2.imshow("The Thresholding Picture", self.image_thresholding)
            cv2.waitKey()
        elif(self.image_nearestInterp_ok==True):
            cv2.imshow("The origin Picture",self.image)
            cv2.imshow("The nearestInterp Picture",self.image_nearestInterp)
            cv2.waitKey()
        elif(self.image_bilinearInterp_ok==True):
            cv2.imshow("The origin Picture", self.image)
            cv2.imshow("The bilinearInterp Picture", self.image_bilinearInterp)
            cv2.waitKey()
        elif(self.image_histEqualized_ok==True):
            cv2.imshow("The gray Picture", self.image_gray)
            cv2.imshow("The histEqualized Picture", self.image_histEqualization)
            cv2.waitKey()
    #显示结果
    def pltshow(self):
        if (self.image_gray_ok==True):
            plt.subplot(1,2,1)
            plt.imshow(self.image_plt)
            plt.subplot(1,2,2)
            plt.imshow(self.image_gray_plt,cmap="gray")
            plt.show()
        elif(self.image_thresholding_ok==True):
            plt.subplot(1, 2, 1)
            plt.imshow(self.image_plt)
            plt.subplot(1, 2, 2)
            plt.imshow(self.image_thresholding_plt,cmap="gray")
            plt.show()
        elif(self.image_nearestInterp_ok==True):
            plt.subplot(1, 2, 1)
            plt.imshow(self.image_plt)
            plt.subplot(1, 2, 2)
            plt.imshow(self.image_nearestInterp_plt)
            plt.show()
        elif(self.image_bilinearInterp_ok==True):
            plt.subplot(1, 2, 1)
            plt.imshow(self.image_plt)
            plt.subplot(1, 2, 2)
            plt.imshow(self.image_bilinearInterp_plt)
            plt.show()
        elif(self.image_histogram_ok==True):
            plt.subplot(1,2,1)
            plt.hist(self.image_gray.ravel(),256,[0,256])
            plt.subplot(1, 2, 2)
            plt.plot(self.gray_count)
            plt.show()
        elif(self.image_histEqualized_ok==True):
            print("Please use the cv2show() function")
if __name__=="__main__":
    imagePath="D:\BaDou\cv\weekTwo\study files\lenna.png"
    imageProcess=ImageProcess(imagePath)
    # imageProcess.handleGray()
    # imageProcess.gray()
    # imageProcess.thresholding()
    # imageProcess.cv2show()
    # imageProcess.thresholdingPlt()
    # imageProcess.pltshow()
    # imageProcess.nearestInterp(600,600)
    # imageProcess.pltshow()
    # imageProcess.cv2show()
    # imageProcess.bilinearInterp(700,700)
    # imageProcess.cv2show()
    # imageProcess.pltshow()
    # imageProcess.handleThresholding()
    # imageProcess.cv2show()
    # imageProcess.pltshow()
    # imageProcess.handleHistogram()
    # imageProcess.pltshow()
    # imageProcess.histogram()
    # imageProcess.pltshow()
    # imageProcess.handleHistEqualization()
    # imageProcess.pltshow()
    # imageProcess.cv2show()
    imageProcess.histEqualization()
    imageProcess.cv2show()
    imageProcess.pltshow()
